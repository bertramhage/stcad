import os
import requests
import zipfile
import io
from datetime import date, timedelta
from argparse import ArgumentParser

def download_ais_data(from_date: date, to_date: date, destination_path: str):
    """
    Downloads and unzips AIS data for a given date range.
    
    AIS data from the Danish Maritime Authority can be found at: http://aisdata.ais.dk/
    
    Args:
        from_date (date): The start date (inclusive).
        to_date (date): The end date (inclusive).
        destination_path (str): The folder path to save the unzipped files.
    """
    
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    base_url = "http://aisdata.ais.dk/"
    current_date = from_date
    
    errors = []
    successes = 0
    while current_date <= to_date:
        year = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        day = current_date.strftime("%d")
        
        # Construct the file name and URL
        file_name = f"aisdk-{year}-{month}-{day}.zip"
        file_url = f"{base_url}{year}/{file_name}"
        
        print(f"Downloading: {file_url}")
        
        try:
            response = requests.get(file_url, stream=True)
            
            if response.status_code == 200:
                with io.BytesIO(response.content) as zip_buffer:
                    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                        zip_ref.extractall(destination_path)
                        unzipped_files = zip_ref.namelist()
                        assert len(unzipped_files) == 1, "Expected exactly one file in the zip archive."
                        successes += 1
                        
            elif response.status_code == 404:
                print(f"Data not found for {current_date} (404 Error). Skipping.")
                errors.append((current_date, "404 Not Found"))
            else:
                print(f"Failed to download {file_name}. Status code: {response.status_code}")
                errors.append((current_date, f"HTTP {response.status_code}"))
                
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during download for {current_date}: {e}")
            errors.append((current_date, str(e)))
        except zipfile.BadZipFile:
            print(f"Downloaded file for {current_date} is not a valid zip file.")
            errors.append((current_date, "Bad Zip File"))
        except AssertionError as ae:
            print(f"Assertion error for {current_date}: {ae}")
            errors.append((current_date, str(ae)))
        
        current_date += timedelta(days=1)

    if len(errors) == 0:
        print("\nAll files downloaded successfully.")
    else:
        print(f"\nDownload succeeded for {successes}/{(successes + len(errors))} days.")
        print(f"Errors encountered for the following dates:")
        for err_date, err_msg in errors:
            print(f" - {err_date}: {err_msg}")
    print("End of download process.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Download AIS data from the Danish Maritime Authority.")
    parser.add_argument("--from_date", type=str, required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--to_date", type=str, required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--destination_path", type=str, required=True, help="Destination folder to save the unzipped files.")
    args = parser.parse_args()
    
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)
    download_ais_data(from_date, to_date, args.destination_path)