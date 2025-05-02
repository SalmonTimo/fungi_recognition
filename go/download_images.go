package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/davidbyttow/govips/v2/vips"
	"golang.org/x/net/http2"
)

// record holds one CSV row
type record struct {
	URL    string
	GBIFID int64
	Genus  string
}

// zarrayMeta describes Zarr array metadata, with compressor
type zarrayMeta struct {
	ZarrFormat int         `json:"zarr_format"`
	Shape      []int       `json:"shape"`
	Chunks     []int       `json:"chunks"`
	Dtype      string      `json:"dtype"`
	Compressor interface{} `json:"compressor"`
	FillValue  interface{} `json:"fill_value"`
	Order      string      `json:"order"`
	Filters    interface{} `json:"filters"`
}

// imageTask is a download job
type imageTask struct {
	idx int
	rec record
}

// imageResult returns processed bytes
type imageResult struct {
	idx  int
	data []byte
	err  error
	rec  record
}

func main() {
	// Initialize libvips
	vips.Startup(nil)
	vips.LoggingSettings(nil, vips.LogLevelWarning)
	defer vips.Shutdown()

	// CLI flags
	csvPath := flag.String("csv", "", "CSV input (identifier,gbifID,genus)")
	outPath := flag.String("out", "", "Output Zarr directory")
	sampleSize := flag.Int("sample-size", 0, "If >0, randomly sample that many rows")
	batchSize := flag.Int("batch-size", 1000, "Chunk size along axis 0")
	concurrency := flag.Int("concurrency", 100, "Max simultaneous downloads")
	flag.Parse()
	if *csvPath == "" || *outPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: download_images -csv file.csv -out data.zarr [--sample-size N]")
		os.Exit(1)
	}

	// Read CSV
	recs, err := readCSV(*csvPath, *sampleSize)
	must(err)
	N := len(recs)

	// Prepare output directory and Zarr metadata with compression
	must(os.MkdirAll(*outPath, 0755))
	writeJSON(filepath.Join(*outPath, ".zgroup"), map[string]int{"zarr_format": 2})

	maxGenus := maxGenusLen(recs)
	compressor := map[string]interface{}{ // Blosc LZ4 compressor
		"id":      "blosc",
		"cname":   "lz4",
		"clevel":  5,
		"shuffle": 1,
	}
	makeArray(*outPath, "images", []int{N, 3, 256, 256}, []int{*batchSize, 3, 256, 256}, "<u1", 0, compressor)
	makeArray(*outPath, "gbifID", []int{N}, []int{*batchSize}, "<i8", 0, nil)
	makeArray(*outPath, "genus", []int{N}, []int{*batchSize}, fmt.Sprintf("|S%%d", maxGenus), "", nil)

	// HTTP client with HTTP/2, keep-alive, gzip
	client := newHTTPClient(*concurrency)

	// Worker pool
	tasks := make(chan imageTask, *concurrency)
	results := make(chan imageResult, *concurrency)
	var wg sync.WaitGroup
	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go worker(client, tasks, results, &wg)
	}

	// Process in batches
	totalBatches := (N + *batchSize - 1) / *batchSize
	for bi := 0; bi < totalBatches; bi++ {
		start := bi * *batchSize
		end := start + *batchSize
		if end > N {
			end = N
		}
		batch := recs[start:end]

		// Dispatch tasks
		for i, rec := range batch {
			tasks <- imageTask{idx: i, rec: rec}
		}

		// Collect results
		imgBuf := make([][]byte, len(batch))
		gbifBuf := make([]int64, len(batch))
		genusBuf := make([]string, len(batch))
		for i := 0; i < len(batch); i++ {
			res := <-results
			if res.err != nil {
				imgBuf[res.idx] = make([]byte, 3*256*256)
			} else {
				imgBuf[res.idx] = res.data
			}
			gbifBuf[res.idx] = res.rec.GBIFID
			genusBuf[res.idx] = res.rec.Genus
		}

		// Write chunk
		writeImagesChunk(*outPath, bi, imgBuf)
		writeGBIFChunk(*outPath, bi, gbifBuf)
		writeGenusChunk(*outPath, bi, genusBuf, maxGenus)
		fmt.Printf("%s Completed batch %d/%d\n", time.Now().Format("15:04:05"), bi+1, totalBatches)
	}

	// Clean up
	close(tasks)
	wg.Wait()
	fmt.Println("Zarr store built at", *outPath)
}

// readCSV loads records and optional sampling
func readCSV(path string, sampleSize int) ([]record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rs, err := csv.NewReader(bufio.NewReader(f)).ReadAll()
	if err != nil {
		return nil, err
	}

	recs := make([]record, 0, len(rs)-1)
	for i, row := range rs {
		if i == 0 {
			if _, err := strconv.ParseInt(row[1], 10, 64); err != nil {
				continue
			}
		}
		gb, _ := strconv.ParseInt(row[1], 10, 64)
		recs = append(recs, record{URL: row[3], GBIFID: gb, Genus: row[2]})
	}
	if sampleSize > 0 && sampleSize < len(recs) {
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(recs), func(i, j int) { recs[i], recs[j] = recs[j], recs[i] })
		recs = recs[:sampleSize]
	}
	return recs, nil
}

// maxGenusLen finds the longest genus string
func maxGenusLen(recs []record) int {
	max := 0
	for _, r := range recs {
		if len(r.Genus) > max {
			max = len(r.Genus)
		}
	}
	return max
}

// newHTTPClient builds an HTTP2-capable client with keep-alive and gzip
func newHTTPClient(concurrency int) *http.Client {
	dialer := &net.Dialer{Timeout: 5 * time.Second}
	tr := &http.Transport{
		DialContext:         dialer.DialContext,
		MaxIdleConns:        concurrency * 2,
		MaxIdleConnsPerHost: concurrency,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  false,
	}
	http2.ConfigureTransport(tr)
	return &http.Client{Transport: tr}
}

// worker downloads and processes one image per task
func worker(client *http.Client, tasks <-chan imageTask, results chan<- imageResult, wg *sync.WaitGroup) {
	defer wg.Done()
	for task := range tasks {
		req, _ := http.NewRequest("GET", task.rec.URL, nil)
		req.Header.Set("Accept-Encoding", "gzip")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		req = req.WithContext(ctx)
		resp, err := client.Do(req)
		if err != nil {
			cancel()
			results <- imageResult{idx: task.idx, data: nil, err: err, rec: task.rec}
			continue
		}
		extractor := resp.Body
		if resp.Header.Get("Content-Encoding") == "gzip" {
			extractor, _ = gzip.NewReader(resp.Body)
		}
		data, err := ioutil.ReadAll(extractor)
		extractor.Close()
		resp.Body.Close()
		cancel()

		if err != nil {
			results <- imageResult{idx: task.idx, data: nil, err: err, rec: task.rec}
			continue
		}
		processed := processImage(data)
		results <- imageResult{idx: task.idx, data: processed, err: nil, rec: task.rec}
	}
}

// processImage decodes with libvips and resizes to 256×256 raw RGB bytes
func processImage(data []byte) []byte {
	img, err := vips.NewImageFromBuffer(data)
	if err != nil {
		return make([]byte, 3*256*256)
	}
	defer img.Close()

	_ = img.AutoRotate()
	w, h := img.Width(), img.Height()
	scale := float64(256) / float64(max(w, h))
	if err := img.Resize(scale, vips.KernelAuto); err != nil {
		return make([]byte, 3*256*256)
	}

	// Only RGB images
	if img.Bands() != 3 {
		return make([]byte, 3*256*256)
	}

	raw, meta, err := img.ExportNative()
	if err != nil {
		return make([]byte, 3*256*256)
	}
	width, height := meta.Width, meta.Height
	buf := raw

	// Compute the actual rowstride Govips used:
	// rowstride = len(raw) / height
	// (it should be >= width*3, rounding up to alignment)
	rowstride := len(buf) / height

	const dim = 256
	canvas := make([]byte, 3*dim*dim)

	// Compute crop/pad offsets
	cropX, cropY := 0, 0
	if width > dim {
		cropX = (width - dim) / 2
	}
	if height > dim {
		cropY = (height - dim) / 2
	}
	padX, padY := 0, 0
	if width < dim {
		padX = (dim - width) / 2
	}
	if height < dim {
		padY = (dim - height) / 2
	}

	// Fill the 256×256 canvas
	for y := 0; y < dim; y++ {
		for x := 0; x < dim; x++ {
			sx := x + cropX - padX
			sy := y + cropY - padY
			if sx < 0 || sy < 0 || sx >= width || sy >= height {
				continue
			}
			// Compute source base index using rowstride:
			srcRowStart := sy * rowstride
			srcIdx := srcRowStart + sx*3
			dstIdx := (y*dim + x) * 3

			// Make sure we never read past buf:
			if srcIdx+2 < len(buf) {
				copy(canvas[dstIdx:dstIdx+3], buf[srcIdx:srcIdx+3])
			}
		}
	}

	return canvas
}

// makeArray writes Zarr metadata
func makeArray(root, name string, shape, chunks []int, dtype string, fill interface{}, compressor interface{}) {
	d := filepath.Join(root, name)
	must(os.MkdirAll(d, 0755))
	meta := zarrayMeta{
		ZarrFormat: 2,
		Shape:      shape,
		Chunks:     chunks,
		Dtype:      dtype,
		Compressor: compressor,
		FillValue:  fill,
		Order:      "C",
		Filters:    nil,
	}
	writeJSON(filepath.Join(d, ".zarray"), meta)
}

// writeImagesChunk writes image bytes to Zarr chunk
func writeImagesChunk(root string, batch int, buf [][]byte) {
	key := fmt.Sprintf("%d.0.0.0", batch)
	fpath := filepath.Join(root, "images", key)
	f, _ := os.Create(fpath)
	defer f.Close()
	for _, b := range buf {
		f.Write(b)
	}
}

// writeGBIFChunk writes int64 IDs
func writeGBIFChunk(root string, batch int, buf []int64) {
	key := fmt.Sprintf("%d", batch)
	fpath := filepath.Join(root, "gbifID", key)
	f, _ := os.Create(fpath)
	defer f.Close()
	for _, v := range buf {
		binary.Write(f, binary.LittleEndian, v)
	}
}

// writeGenusChunk writes fixed-width genus strings
func writeGenusChunk(root string, batch int, buf []string, maxLen int) {
	key := fmt.Sprintf("%d", batch)
	fpath := filepath.Join(root, "genus", key)
	f, _ := os.Create(fpath)
	defer f.Close()
	for _, s := range buf {
		b := []byte(s)
		if len(b) < maxLen {
			b = append(b, bytes.Repeat([]byte{0}, maxLen-len(b))...)
		} else if len(b) > maxLen {
			b = b[:maxLen]
		}
		f.Write(b)
	}
}

// writeJSON writes a JSON file
func writeJSON(path string, v interface{}) {
	b, _ := json.MarshalIndent(v, "", "  ")
	must(ioutil.WriteFile(path, b, 0644))
}

// must panics on error
func must(err error) {
	if err != nil {
		panic(err)
	}
}

// helper for max of two ints
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
