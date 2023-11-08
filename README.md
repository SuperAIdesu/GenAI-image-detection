# AI-generated image detection

## Managing Python packages

Use of `pipenv` is recommended. The required packages are in `Pipfile`, and can be installed using `pipenv install`.

## Scraping script for Reddit

`python scrape.py --subreddit midjourney --flair Showcase`

This command will scrape the midjourney subreddit, and filter posts that contain the "Showcase" flair. The default number of images to scrape is 30000. The output will contain a parquet file containing metadata, and a csv file containing the urls.

`img2dataset --url_list=urls/midjourney.csv --output_folder=data/midjourney --thread_count=64 --resize_mode=no --output_format=webdataset`

This command will download the images in the webdataset format.


## Laion script for real images

`wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .`

This command will download a 50GB url metadata dataset in 32 parquet files.

`sample_laion_script.ipynb`

This script consolidates the parquet files, excludes NSFW images, and selects a subset of 224,917 images.

`combine_laion_script`

This script combines the outputs from earlier into 1 parquet file.

`img2dataset --url_list urls/laion.parquet --input_format "parquet" --url_col "URL" --caption_col "TEXT" --skip_reencode True --output_format webdataset --output_folder data/laion400m_data --processes_count 16 --thread_count 128 --resize_mode no --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb True`

This command will download the images in the webdataset format.
