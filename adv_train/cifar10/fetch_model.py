from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse

parser = argparse.ArgumentParser(description='fetch pretrained model')
parser.add_argument('--method', type=str, default="spade",
        help="choose training method, [spade, random, pgd, clean]")


args = parser.parse_args()
clean = "1Tolc5n6W1ARBY9nCVjzp-828r1i502g_"
pgd = "1-_zFmM9QsKFur76STMbMUhRTK_oMDPnp"
spade = "11onZAKx4t9avFRyKi2gyUIU3FBC8zuMF"
random = "1Gwa59RuJ5QtY3n1_mecSC97JmIbwzSCZ"

if args.method == "clean":
    fid = clean
    zip_name = "pgd_0.0.zip"
elif args.method == "pgd":
    fid = pgd
    zip_name = "pgd_8.0.zip"
elif args.method == "spade":
    fid = spade
    zip_name = "pgd-spade_6.0_8.0.zip"
elif args.method == "random":
    fid = random
    zip_name = "pgd-random_6.0_8.0.zip"

gdd.download_file_from_google_drive(file_id=fid,
                                            dest_path="./models/{}".format(zip_name),
                                            unzip=True)
