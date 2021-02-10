from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse

parser = argparse.ArgumentParser(description='fetch pretrained model')
parser.add_argument('--method', type=str, default="spade",
        help="choose training method, [spade, random, pgd, clean]")


args = parser.parse_args()
clean = "14bkJ9yqFmlQddkcH9b3RIPIW7V-dne8_"
pgd = "18J_AQiBRXyZ0qXDHgXrgl0iScKKiSjCJ"
spade = "1wE4MhkLzx44N3Q_bmOBDLl2iYYhnconc"
random = "1_smN2BNQCHTaRdbL0Nfgds_fCm6eF_Nb"

if args.method == "clean":
    fid = clean
    zip_name = "pgd_0.0.zip"
elif args.method == "pgd":
    fid = pgd
    zip_name = "pgd_0.3.zip"
elif args.method == "spade":
    fid = spade
    zip_name = "pgd-spade_0.2_0.3.zip"
elif args.method == "random":
    fid = random
    zip_name = "pgd-random_0.2_0.3.zip"

gdd.download_file_from_google_drive(file_id=fid,
                                            dest_path="./models/{}".format(zip_name),
                                            unzip=True)
