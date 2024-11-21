from src.options import config_settings

def main():
    opt = config_settings()
    print(opt)
    # data = fetch_data_from_api()
    # process_data(data)

if __name__ == "__main__":
    main()