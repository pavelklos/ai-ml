# Getting the NIH Chest X-ray Dataset

The NIH Chest X-ray Dataset is a large public dataset containing over 100,000 chest X-ray images from more than 30,000 patients, with 14 disease labels. Here's how you can obtain it:

## Option 1: Download from the Official Source (NIH)

You can download the dataset from the NIH's official website:

1. Visit the NIH Clinical Center page: https://nihcc.app.box.com/v/ChestXray-NIHCC

2. You'll find several download links:
   - Images split into 12 parts (approximately 7-8GB each)
   - A Data_Entry_2017.csv file (metadata file with labels)
   - BBox_List_2017.csv (for bounding box annotations)
   
3. Download the metadata CSV file first and then the image files, which are split into multiple parts (images_00*.tar.gz).

```bash
# Example commands to download and extract
mkdir -p NIH_Chest_X_rays/images
cd NIH_Chest_X_rays

# Download metadata
wget https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789944

# Download image parts (this will take time)
for i in {1..12}; do
    num=$(printf "%03d" $i)
    wget https://nihcc.app.box.com/v/ChestXray-NIHCC/file/images_${num}.tar.gz
    tar -xzf images_${num}.tar.gz -C images/
    rm images_${num}.tar.gz  # Optional: remove archive after extraction
done
```

## Option 2: Download from Kaggle

The dataset is also available on Kaggle, which might be easier to download using the Kaggle API:

1. Make sure you have the Kaggle API set up (install with `pip install kaggle`)
2. Configure your Kaggle API credentials
3. Download the dataset:

```bash
# Create directory structure
mkdir -p NIH_Chest_X_rays/images

# Download the dataset from Kaggle
kaggle datasets download -d nih-chest-xrays/data
unzip data.zip -d NIH_Chest_X_rays
```

## Important Notes

1. **Size considerations**: The complete dataset is approximately 45-50GB. Your M1 Max with 64GB RAM should handle it well, but you might want to start with a subset.

2. **Directory structure**: The code you shared expects this structure:
   ```
   NIH_Chest_X_rays/
   ├── images/            # All image files (.png)
   └── Data_Entry_2017.csv  # Metadata file with labels
   ```

3. **Using a subset for testing**: If you want to test your code before downloading the entire dataset, you can use a smaller subset:

```python
# After loading the CSV, select a smaller subset
df = pd.read_csv(metadata_path)
small_df = df.sample(1000, random_state=42)  # Work with just 1000 images for testing
```

4. **Alternative: Using Kaggle notebooks**: Kaggle provides free GPU notebooks with this dataset pre-loaded, which might be a good way to start without downloading everything.