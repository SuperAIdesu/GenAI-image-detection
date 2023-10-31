# AI-generated image detection

## Managing Python packages

Use of `pipenv` is recommended. The required packages are in `Pipfile`, and can be installed using `pipenv install`.

## Scraping script for Reddit

`python scrape.py --subreddit midjourney --flair Showcase`

This command will scrape the midjourney subreddit, and filter posts that contain the "Showcase" flair. The default number of images to scrape is 30000. The output will contain a parquet file containing metadata, and a csv file containing the urls.

`img2dataset --url_list=urls/midjourney.csv --output_folder=data/midjourney --thread_count=64 --resize_mode=no --output_format=webdataset`

This command will download the images in the webdataset format.