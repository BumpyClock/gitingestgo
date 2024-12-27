package main

import (
	"bufio"
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/mr-tron/base58/base58"
	"github.com/pkoukk/tiktoken-go"
)

const (
	// MaxFileSize indicates the maximum size of a file (in bytes) to be processed.
	MaxFileSize = 10 * 1024 * 1024 // 10 MB

	// MaxDirectoryDepth sets the maximum directory depth to scan to avoid runaway recursion.
	MaxDirectoryDepth = 20

	// MaxFiles sets the maximum number of files to process before stopping.
	MaxFiles = 10000

	// MaxTotalSizeBytes indicates the maximum total size of files (in bytes) before stopping the scan.
	MaxTotalSizeBytes = 500 * 1024 * 1024 // 500 MB

	// DefaultIgnoreFile is the default file (e.g. .gitignore) used for ignore patterns, if not overridden.
	DefaultIgnoreFile = ".gitignore"
)

// FileInfo holds essential file data, including path, content, and size.
type FileInfo struct {
	Path    string
	Content string
	Size    int64
}

/*
parseIgnoreFile reads lines from an ignore file (e.g. .gitignore) and returns the
patterns to be ignored. Lines starting with '#' or empty lines are skipped.
This function is used to ensure no unneeded files pollute our scan results.
*/
func parseIgnoreFile(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		// Return an error if the ignore file can't be opened
		return nil, fmt.Errorf("unable to open ignore file: %w", err)
	}
	defer func() {
		_ = file.Close()
	}()

	var patterns []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			// Skip comments and empty lines
			continue
		}
		patterns = append(patterns, line)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading ignore file: %w", err)
	}

	return patterns, nil
}

/*
matchesPattern checks if a given path (relative to the root) matches a .gitignore pattern.
- It handles negation (!) patterns.
- It recognizes patterns like "*" (recursive) and "/" (root-based) for directories.
- If a file or directory is matched by a pattern, it is considered ignored.
*/
func matchesPattern(path string, pattern string, isDir bool) bool {
	negate := false
	if strings.HasPrefix(pattern, "!") {
		negate = true
		pattern = pattern[1:]
	}

	// Remove leading slash in patterns to standardize
	if strings.HasPrefix(pattern, "/") {
		pattern = pattern[1:]
	}

	// Handle '**/' for recursive directory matching
	if strings.Contains(pattern, "**/") {
		// Convert to a more general regular expression
		pattern = strings.ReplaceAll(pattern, "**/", "(.*/)?")
	}

	// Directory-specific pattern: If pattern ends with '/', it applies only to directories
	if strings.HasSuffix(pattern, "/") {
		if isDir {
			pattern = strings.TrimSuffix(pattern, "/") + "(/.*)?"
		} else {
			return false
		}
	}

	// Convert wildcard expressions to regular expressions
	pattern = "^" + strings.ReplaceAll(pattern, ".", "\\.")
	pattern = strings.ReplaceAll(pattern, "*", ".*")
	pattern = pattern + "$"

	matched, err := regexp.MatchString(pattern, path)
	if err != nil {
		// If there's an invalid pattern, treat it as non-matching for safety
		log.Printf("warning: invalid pattern encountered: %v", err)
		return false
	}

	return matched == !negate
}

/*
isTextFile uses heuristic checks to determine if a file is textual:
  - Checks for Unicode Byte Order Mark (BOM).
  - Searches for null bytes or high-range control characters that typically
    do not appear in text files.

This helps avoid reading binary files which can disrupt token counting or logging.
*/
func isTextFile(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		// We cannot open the file, so skip
		log.Printf("warning: could not open file '%s': %v", filename, err)
		return false
	}
	defer func() {
		_ = f.Close()
	}()

	// Read up to 4 bytes to detect BOM
	bom := make([]byte, 4)
	if _, err := f.Read(bom); err == nil {
		if isUnicodeBOM(bom) {
			return true
		}
	}

	// Reset file pointer after BOM check
	_, _ = f.Seek(0, io.SeekStart)

	// Check for null bytes in the first ~1KB
	buf := make([]byte, 1024)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		log.Printf("warning: could not read from file '%s': %v", filename, err)
		return false
	}
	for _, b := range buf[:n] {
		// Null bytes or unusual control characters typically indicate binary data
		if b == 0 || (b < 32 && b != 9 && b != 10 && b != 13) {
			return false
		}
	}
	return true
}

/*
isUnicodeBOM returns true if the first few bytes of the file represent
any recognized Unicode BOM (UTF-8, UTF-16, UTF-32 variants).
*/
func isUnicodeBOM(bom []byte) bool {
	return (bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF) || // UTF-8
		(bom[0] == 0xFE && bom[1] == 0xFF) || // UTF-16 (BE)
		(bom[0] == 0xFF && bom[1] == 0xFE) || // UTF-16 (LE)
		(bom[0] == 0x00 && bom[1] == 0x00 && bom[2] == 0xFE && bom[3] == 0xFF) || // UTF-32 (BE)
		(bom[0] == 0xFF && bom[1] == 0xFE && bom[2] == 0x00 && bom[3] == 0x00) // UTF-32 (LE)
}

/*
readFileLimited reads the entire content of a file up to a specified maxFileSize.
If the file size exceeds maxFileSize, reading is aborted to save resources.
*/
func readFileLimited(path string, maxFileSize int64) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open file '%s': %w", path, err)
	}
	defer func() {
		_ = f.Close()
	}()

	fi, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to get file info '%s': %w", path, err)
	}
	if fi.Size() > maxFileSize {
		return "", fmt.Errorf("file '%s' exceeds max file size limit", path)
	}

	var builder strings.Builder
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		builder.WriteString(scanner.Text())
		builder.WriteString("\n")
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error scanning file '%s': %w", path, err)
	}

	return builder.String(), nil
}

/*
appendToTree is a utility for building a "directory tree" representation.
The relPath is the path relative to the basePath. We count the number of
separators to derive indentation. We add ASCII tree prefixes like "├──", "└──", etc.
*/
func appendToTree(builder *strings.Builder, relPath string, basePath string, isDir bool) {
	depth := strings.Count(relPath, string(os.PathSeparator))

	// Direct child of the root
	if depth == 0 {
		if isDir {
			builder.WriteString(relPath + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(relPath + "\n")
		}
		return
	}

	// For deeper paths, we build a prefix for indentation
	prefix := strings.Repeat("│   ", depth-1)
	isLast := isLastElement(basePath, relPath)
	name := filepath.Base(relPath)

	if isLast {
		if isDir {
			builder.WriteString(prefix + "└── " + name + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(prefix + "└── " + name + "\n")
		}
	} else {
		if isDir {
			builder.WriteString(prefix + "├── " + name + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(prefix + "├── " + name + "\n")
		}
	}
}

/*
isLastElement checks directory contents to see if the current file/directory
is the last one at its depth level. This ensures correct usage of "├──" vs. "└──"
in the ASCII tree output.
*/
func isLastElement(basePath, relPath string) bool {
	parts := strings.Split(relPath, string(os.PathSeparator))
	if len(parts) < 2 {
		// Root or single-level element is always considered "last" since there's no sibling
		return true
	}

	parentDir := filepath.Join(basePath, strings.Join(parts[:len(parts)-1], string(os.PathSeparator)))
	currentBase := parts[len(parts)-1]

	entries, err := os.ReadDir(parentDir)
	if err != nil {
		// If we can't read the directory, assume it's last to avoid repeated errors
		log.Printf("warning: could not read parent directory '%s': %v", parentDir, err)
		return true
	}

	// Collect names in the parent directory
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}

	// If the last item in alphabetical order matches the current file name, it's last
	return len(names) > 0 && names[len(names)-1] == currentBase
}

/*
formatFilesContent organizes the file contents with a clear heading separator.
- If a README.md file exists, it’s moved to the top to prioritize user documentation.
- After that, each file’s content is displayed with a "File: {filename}" header.
*/
func formatFilesContent(files []FileInfo) string {
	var contentBuilder strings.Builder

	// We use a separator for clarity between files
	separator := strings.Repeat("=", 48) + "\n"

	// If there's a README, place it first
	for i, file := range files {
		if strings.ToLower(file.Path) == "readme.md" {
			contentBuilder.WriteString(separator)
			contentBuilder.WriteString(fmt.Sprintf("File: %s\n", file.Path))
			contentBuilder.WriteString(separator)
			contentBuilder.WriteString(file.Content + "\n\n")

			// Remove README from the slice so we don't duplicate below
			files = append(files[:i], files[i+1:]...)
			break
		}
	}

	// Append the rest of the files
	for _, file := range files {
		contentBuilder.WriteString(separator)
		contentBuilder.WriteString(fmt.Sprintf("File: %s\n", file.Path))
		contentBuilder.WriteString(separator)
		contentBuilder.WriteString(file.Content + "\n\n")
	}

	return contentBuilder.String()
}

/*
countTokens uses the TikToken library to estimate the number of tokens in
the provided text. We return a user-friendly formatted token count (e.g., "1.2k").
*/
func countTokens(text string) (string, error) {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return "", fmt.Errorf("failed to get encoding: %w", err)
	}

	totalTokens := len(enc.Encode(text, nil, nil))
	switch {
	case totalTokens > 1_000_000:
		return fmt.Sprintf("%.1fM", float64(totalTokens)/1_000_000), nil
	case totalTokens > 1_000:
		return fmt.Sprintf("%.1fk", float64(totalTokens)/1_000), nil
	default:
		return fmt.Sprintf("%d", totalTokens), nil
	}
}

/*
createSummary prepares a brief summary of the scan results:
- Repository name
- Number of files analyzed
- Subpath (if specified)
- Commit or branch (if specified)
- Estimated tokens from the text content
*/
func createSummary(repoName, branch, commit, subpath string, fileCount int, tokenCount string) string {
	var summary strings.Builder
	summary.WriteString(fmt.Sprintf("Repository: %s\n", repoName))
	summary.WriteString(fmt.Sprintf("Files analyzed: %d\n", fileCount))

	if subpath != "" && subpath != "/" {
		summary.WriteString(fmt.Sprintf("Subpath: %s\n", subpath))
	}
	if commit != "" {
		summary.WriteString(fmt.Sprintf("Commit: %s\n", commit))
	} else if branch != "" && branch != "main" && branch != "master" {
		summary.WriteString(fmt.Sprintf("Branch: %s\n", branch))
	}

	summary.WriteString(fmt.Sprintf("Estimated tokens: %s", tokenCount))

	return summary.String()
}

/*
parsePatterns takes a comma-separated string of patterns, trims spaces, and
returns a slice for each pattern. This is used to handle user-supplied
exclude patterns.
*/
func parsePatterns(patterns string) []string {
	var parsedPatterns []string
	for _, p := range strings.Split(patterns, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			parsedPatterns = append(parsedPatterns, p)
		}
	}
	return parsedPatterns
}

/*
overrideIgnorePatterns ensures that any explicitly included patterns
(in `includePatterns`) are not excluded by default. This is done by filtering out
patterns that match the included ones.
*/
func overrideIgnorePatterns(ignorePatterns []string, includePatterns []string) []string {
	var newIgnorePatterns []string
	for _, ignorePattern := range ignorePatterns {
		shouldIgnore := true
		for _, includePattern := range includePatterns {
			matched, _ := filepath.Match(includePattern, ignorePattern)
			if matched {
				shouldIgnore = false
				break
			}
		}
		if shouldIgnore {
			newIgnorePatterns = append(newIgnorePatterns, ignorePattern)
		}
	}
	return newIgnorePatterns
}

/*
parseIncludePatterns normalizes user-provided include patterns:
- Removes leading './' or '/'.
- If the pattern ends with '/', we add '**' to match any subdirectories or files under it.
*/
func parseIncludePatterns(pattern string) []string {
	parts := strings.Split(strings.TrimSpace(pattern), ",")
	var patterns []string

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if strings.HasPrefix(part, "./") {
			part = part[2:]
		} else if strings.HasPrefix(part, "/") {
			part = part[1:]
		}
		if strings.HasSuffix(part, "/") {
			patterns = append(patterns, part+"**")
		} else {
			patterns = append(patterns, part)
		}
	}

	return patterns
}

/*
ParseQuery orchestrates the input parsing:
- Validates whether the input is a local path vs. a web URL (web URLs are unsupported here).
- Sets default ignore patterns and overrides them with user-supplied excludes/includes.
- Creates a short "id" from hashing the absolute path to keep track of the project reference.
Returns a map that consolidates all relevant query metadata.
*/
func ParseQuery(input string, maxFileSizeKB int, fromWeb bool, includePatternsStr, excludePatternsStr string) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	if fromWeb {
		// For demonstration, we return an explicit error here if fromWeb is true
		return nil, errors.New("web URL processing is not implemented for CLI in this example")
	}

	// Validate local path and convert to absolute
	absPath, err := filepath.Abs(input)
	if err != nil {
		return nil, fmt.Errorf("invalid local path: %w", err)
	}

	// Basic input validation: ensure path actually exists
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("the specified path does not exist: %s", absPath)
	}

	// Populate result with key-value pairs
	result["local_path"] = absPath
	result["slug"] = filepath.Base(absPath) // A short project slug based on the directory name
	result["subpath"] = "/"

	// Generate a short ID from path-based hashing
	idBytes := sha256.Sum256([]byte(absPath))
	result["id"] = base58.Encode(idBytes[:16]) // Keep first 16 bytes for brevity

	// Default ignore patterns for typical file types that are not relevant to code analysis
	defaultIgnorePatterns := []string{
		"*.o", "*.exe", "*.dll", "*.so", "*.dylib",
		"*.pyc", "*.pyo", "*.pyd", "*.class", "*.jar", "*.war", "*.ear",
		"*.obj", "*.a", "*.lib",
		"node_modules/", "vendor/", "bower_components/",
		".git/", ".svn/", ".hg/", ".DS_Store/",
		"__pycache__/", "venv/", ".venv/", "env/", ".env/",
		".idea/", ".vscode/", "*.sublime-project", "*.sublime-workspace",
		"*.log", "*.tmp", "*.bak",
		".next/", ".nuxt/", "dist/", "build/", "target/",
		"*.zip", "*.tar", "*.tar.gz", "*.rar", "*.7z",
		"*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.ico",
		"*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.flv",
		"*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg",
		"*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
		".gitignore", ".DS_Store",
		"*.pem", "*.cer", "*.crt", "*.key",
		"go.mod", "go.sum", "package-lock.json", "yarn.lock",
		"*.min.js", "*.min.css",
		"*.woff", "*.woff2", "*.eot", "*.ttf", "*.otf",
		"*.svg", "*.ico", "*.webp",
		"*.json", "*.xml", "*.yaml", "*.yml", "*.toml",
		"*.md", "*.markdown", "*.rst", "*.txt",
		"*.mod", "*.sum", "*.lock",
	}

	var finalIgnorePatterns []string
	finalIgnorePatterns = append(finalIgnorePatterns, defaultIgnorePatterns...)

	// Add user-supplied exclude patterns if present
	if excludePatternsStr != "" {
		excludePatterns := parsePatterns(excludePatternsStr)
		finalIgnorePatterns = append(finalIgnorePatterns, excludePatterns...)
	}

	result["ignore_patterns"] = finalIgnorePatterns

	// Process include patterns if provided, overriding any default ignore that conflicts
	if includePatternsStr != "" {
		includePatterns := parseIncludePatterns(includePatternsStr)
		result["include_patterns"] = includePatterns
		// Filter out default ignores that match the includes
		finalIgnorePatterns = overrideIgnorePatterns(finalIgnorePatterns, includePatterns)
		result["ignore_patterns"] = finalIgnorePatterns
	}

	// Convert KB to bytes
	result["max_file_size"] = int64(maxFileSizeKB * 1024)

	return result, nil
}

/*
scanLocalDirectory scans the directory recursively (or processes a single file),
respecting all ignore patterns, maximum file-size limits, etc.
It returns a list of FileInfo and a formatted directory tree string.
*/
func scanLocalDirectory(
	basePath string,
	ignorePatterns []string,
	maxFileSize int64,
) ([]FileInfo, string, error) {

	var files []FileInfo
	var dirStructure strings.Builder

	// Sync primitives to handle concurrent file reading
	var mu sync.Mutex
	var wg sync.WaitGroup

	fileInfo, err := os.Lstat(basePath)
	if err != nil {
		return nil, "", fmt.Errorf("unable to get file info for '%s': %w", basePath, err)
	}

	isSymlinkToDir := false
	if fileInfo.Mode()&os.ModeSymlink != 0 {
		// If it's a symlink, we resolve it
		realPath, err := filepath.EvalSymlinks(basePath)
		if err != nil {
			return nil, "", fmt.Errorf("failed to evaluate symlink for '%s': %w", basePath, err)
		}
		realFileInfo, err := os.Stat(realPath)
		if err != nil {
			return nil, "", fmt.Errorf("failed to stat symlink target '%s': %w", realPath, err)
		}
		isSymlinkToDir = realFileInfo.IsDir()
	}

	// If it’s a directory (or symlink to a directory), traverse recursively
	if isSymlinkToDir || fileInfo.IsDir() {
		err = filepath.Walk(basePath, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				log.Printf("warning: error during filepath.Walk at '%s': %v", path, walkErr)
				return walkErr
			}

			// Avoid recursing further into symlinked directories to reduce risk of infinite loops
			if info.Mode()&os.ModeSymlink != 0 {
				log.Printf("info: skipping symlinked directory '%s'", path)
				return filepath.SkipDir
			}

			relPath, _ := filepath.Rel(basePath, path)
			if relPath == "." {
				// Skip root directory itself from listing
				return nil
			}

			// Apply ignore patterns before processing
			for _, pattern := range ignorePatterns {
				if matchesPattern(relPath, pattern, info.IsDir()) {
					if info.IsDir() {
						// If a directory matches, skip the entire sub-tree
						log.Printf("info: skipping directory '%s' due to ignore pattern '%s'", relPath, pattern)
						return filepath.SkipDir
					}
					// If a file is ignored, just skip it
					log.Printf("info: skipping file '%s' due to ignore pattern '%s'", relPath, pattern)
					return nil
				}
			}

			if info.IsDir() {
				// Build directory tree
				appendToTree(&dirStructure, relPath, basePath, true)
			} else if isTextFile(path) && info.Size() <= maxFileSize {
				// Append file to tree
				appendToTree(&dirStructure, relPath, basePath, false)

				// Process file in a goroutine to improve parallel reads
				wg.Add(1)
				go func(p string, size int64) {
					defer wg.Done()
					content, fileErr := readFileLimited(p, maxFileSize)
					if fileErr != nil {
						log.Printf("warning: reading file '%s' failed: %v", p, fileErr)
						return
					}
					// Protect concurrent map/slice access
					mu.Lock()
					files = append(files, FileInfo{
						Path:    relPath,
						Content: content,
						Size:    size,
					})
					mu.Unlock()
				}(path, info.Size())
			}

			return nil
		})

	} else {
		// If it's a single file, process it directly (non-directory)
		if isTextFile(basePath) && fileInfo.Size() <= maxFileSize {
			content, readErr := readFileLimited(basePath, maxFileSize)
			if readErr == nil {
				files = append(files, FileInfo{
					Path:    filepath.Base(basePath),
					Content: content,
					Size:    fileInfo.Size(),
				})
				dirStructure.WriteString(filepath.Base(basePath) + "\n")
			} else {
				log.Printf("warning: reading single file '%s' failed: %v", basePath, readErr)
			}
		}
	}

	// Wait for all concurrent file reads to finish
	wg.Wait()

	if err != nil {
		return files, dirStructure.String(), fmt.Errorf("error walking through directory '%s': %w", basePath, err)
	}

	return files, dirStructure.String(), nil
}

/*
IngestFromQuery executes the main scanning logic by reading .gitignore (if exists),
combining ignore patterns, scanning the path, and finally returning:
- A summary of the scan
- The ASCII tree representation of the directory
- The combined file contents for text files
*/
func IngestFromQuery(query map[string]interface{}) (summary string, tree string, filesContent string, err error) {
	localPath := query["local_path"].(string)

	// Parse .gitignore in the local directory if available
	var gitIgnorePatterns []string
	gitIgnorePath := filepath.Join(localPath, DefaultIgnoreFile)
	if _, statErr := os.Stat(gitIgnorePath); statErr == nil {
		if patterns, parseErr := parseIgnoreFile(gitIgnorePath); parseErr == nil {
			gitIgnorePatterns = append(gitIgnorePatterns, patterns...)
		} else {
			log.Printf("info: error parsing '%s': %v", gitIgnorePath, parseErr)
		}
	}

	// Merge custom ignore patterns with .gitignore
	ignorePatterns := []string{}
	if userIgnore, ok := query["ignore_patterns"].([]string); ok {
		ignorePatterns = append(ignorePatterns, userIgnore...)
	}
	ignorePatterns = append(ignorePatterns, gitIgnorePatterns...)

	// Ensure .git/ is always ignored, even if not in .gitignore
	ignorePatterns = append(ignorePatterns, ".git/")

	maxFileSize := query["max_file_size"].(int64)
	files, dirTree, scanErr := scanLocalDirectory(localPath, ignorePatterns, maxFileSize)
	if scanErr != nil {
		return "", "", "", fmt.Errorf("scan error: %w", scanErr)
	}

	// Build the combined file content output
	formattedContent := formatFilesContent(files)

	// Count tokens to provide an estimate
	fullTextForTokenCount := dirTree + formattedContent
	tokenCount, tokenErr := countTokens(fullTextForTokenCount)
	if tokenErr != nil {
		return "", "", "", fmt.Errorf("token counting error: %w", tokenErr)
	}

	// Gather metadata for summary
	repoName, _ := query["slug"].(string)
	branch, _ := query["branch"].(string)
	commit, _ := query["commit"].(string)
	subpath, _ := query["subpath"].(string)

	summaryText := createSummary(repoName, branch, commit, subpath, len(files), tokenCount)

	return summaryText, dirTree, formattedContent, nil
}

func main() {
	/*
	   Command-line Flags:

	   -d string : Directory (or file) to scan
	   -o string : Output file to write combined output
	   -s int    : Max file size in KB (default 10)
	   -i string : Include patterns (comma-separated)
	   -e string : Exclude patterns (comma-separated)

	   Example usage:
	   go run main.go -d ./your_project -o output.txt -s 20 -i "*.go" -e "*.md"
	*/
	inputPtr := flag.String("d", "", "Directory (or file) to process")
	outputPtr := flag.String("o", "", "Output file name")
	maxFileSizePtr := flag.Int("s", 10, "Max file size in KB")
	includePatternsPtr := flag.String("i", "", "Include patterns (comma-separated)")
	excludePatternsPtr := flag.String("e", "", "Exclude patterns (comma-separated)")

	flag.Parse()

	// Basic validation of required flags
	if *inputPtr == "" {
		log.Fatal("error: you must provide an input directory/file using -d <path>")
	}
	if *outputPtr == "" {
		log.Fatal("error: you must provide an output file name using -o <filename>")
	}

	// Step 1: Parse query (respects advanced reasoning for ignoring unneeded content)
	query, err := ParseQuery(
		*inputPtr,
		*maxFileSizePtr,
		false, // fromWeb
		*includePatternsPtr,
		*excludePatternsPtr,
	)
	if err != nil {
		log.Fatalf("error parsing query: %v", err)
	}

	// Step 2: Ingest data (scan, build tree, combine file content)
	summary, tree, content, err := IngestFromQuery(query)
	if err != nil {
		log.Fatalf("error during ingestion: %v", err)
	}

	// Step 3: Write results to the specified output file
	outputContent := "Directory structure:\n" + tree + "\n" + content
	if writeErr := os.WriteFile(*outputPtr, []byte(outputContent), 0644); writeErr != nil {
		log.Fatalf("error writing output to file '%s': %v", *outputPtr, writeErr)
	}

	// Display success message and summary to the console
	fmt.Printf("Successfully processed path: %s\n", *inputPtr)
	fmt.Printf("Output written to: %s\n", *outputPtr)
	fmt.Println(summary)
}
