# GoPython CLI

A simple command-line tool that executes Python files and optionally cleans directories of specific file types. Just a handy little thing that saves me some time on some common enough tasks that I do.   


## Build

```go build -o gopython```

## Usage

```bash
# Run a Python file
./gopython test.py

# Run Python file and clean directory
./gopython test.py ./test_dir

# Run Python file and clean directory, ignoring specific extensions
./gopython -i .csv test.py ./test_dir
./gopython -i ".csv,.txt" test.py ./test_dir

# Run with confirmation prompt for each deletion
./gopython -confirm test.py ./test_dir

# Combine flags
./gopython -confirm -i .csv test.py ./test_dir