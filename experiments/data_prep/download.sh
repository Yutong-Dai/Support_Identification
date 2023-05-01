DATASETS_DIR="$(realpath ~)/db/"
# check dir existence
if [ -d "$DATASETS_DIR" ]; then
    echo "$DATASETS_DIR exists."
else 
    echo "$DATASETS_DIR does not exist. Making one..."
    mkdir -P $DATASETS_DIR
fi

for url in $(cat dataset_url.txt)
do 
  dataset=${url##*/}
  file="${DATASETS_DIR}${dataset}"
  # check a file's existence
  if [ -f "$file" ]; then
      echo "$file exists. Skipping..."
  else 
      echo "$file does not exist."
      wget -P $DATASETS_DIR $url
  fi
done

mkdir -p $DATASETS_DIR/Lip $DATASETS_DIR/lammax
