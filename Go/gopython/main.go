package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	ignoreFlag := flag.String("i", "", "File extensions to ignore (comma-separated)")

	confirmFlag := flag.Bool("confirm", false, "Confirm deletions")

	flag.Parse()

	args := flag.Args()

	if len(args) < 1 {
		fmt.Println("Usage: gopython [-i '.ext1,.ext2'] <python_file> [directory_to_clean]")
		os.Exit(1)
	}

	if len(args) > 1 {
		dirToClean := args[1]

		ignoredExts := []string{}
		if *ignoreFlag != "" {

			for _, ext := range strings.Split(*ignoreFlag, ",") {
				ext = strings.TrimSpace(ext)
				if !strings.HasPrefix(ext, ".") {
					ext = "." + ext
				}
				ignoredExts = append(ignoredExts, strings.ToLower(ext))
			}
			fmt.Printf("Ignoring extensions: %v\n", ignoredExts)
		}

		err := cleanDirectory(dirToClean, ignoredExts, *confirmFlag)
		if err != nil {
			fmt.Printf("Error cleaning directory: %v\n", err)
			os.Exit(1)
		}
	}
	pythonFile := args[0]

	cmd := exec.Command("python", pythonFile)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		fmt.Printf("Error executing Python file: %v\n", err)
		os.Exit(1)
	}

}

func cleanDirectory(dir string, ignoredExts []string, confirm bool) error {

	extensions := []string{".jpg", ".jpeg", ".png", ".gif", ".csv", ".txt"}

	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))

		for _, ignoredExt := range ignoredExts {
			if ext == ignoredExt {
				fmt.Printf("Ignoring: %s\n", path)
				return nil
			}
		}

		for _, targetExt := range extensions {
			if ext == targetExt {
				if confirm {
					fmt.Printf("Delete %s? (y/n): ", path)
					var response string
					_, err := fmt.Scanln(&response)
					if err != nil {
						return err

					}
					response = strings.ToLower(strings.TrimSpace(response))
					if response != "y" && response != "yes" {
						fmt.Printf("Skipping: %s\n", path)
						return nil
					}
				}
				err := os.Remove(path)
				if err != nil {
					return err
				}
				fmt.Printf("Removed: %s\n", path)
				break
			}
		}
		return nil
	})
}
