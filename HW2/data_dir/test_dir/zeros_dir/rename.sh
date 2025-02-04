#!/bin/bash

# Counter to append to the new file names
counter=1

# Loop through the files you want to rename
for file in *; do
  # Check if it's a regular file (not a directory or symbolic link)
  if [ -f "$file" ]; then
    # Get the file extension (if any)
    extension="${file##*.}"

    # Rename the file to the current counter value
    # If the file has an extension, keep it, otherwise rename without extension
    if [[ "$file" == *.* ]]; then
      mv "$file" "$counter.$extension"
    else
      mv "$file" "$counter"
    fi
    
    # Increment the counter
    ((counter++))
  fi
done
