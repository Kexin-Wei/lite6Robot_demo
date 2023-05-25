from UGS_Client_Module import UGSClient

def main():
    reader = UGSClient()
    reader.readImageAndTrackObjects()

if __name__ == "__main__":
    main()