#!/bin/bash

# Usage: ./compare_output.sh <program> <known_output_file>

# Get the program and output file from the command line arguments
PROGRAM=$1
KNOWN_OUTPUT_FILE=$2

# Temporary file to store the program output
TEMP_OUTPUT_FILE=$(mktemp)

# Run the program and redirect its output to the temporary file
$PROGRAM > $TEMP_OUTPUT_FILE

# Compare the temporary output file with the known output file
if diff -q "$TEMP_OUTPUT_FILE" "$KNOWN_OUTPUT_FILE" > /dev/null; then
    echo "pass"
else
    echo "FAIL"
    diff "$TEMP_OUTPUT_FILE" "$KNOWN_OUTPUT_FILE"
fi

# Clean up the temporary file
rm "$TEMP_OUTPUT_FILE"
