import numpy as np
import csv
import os
import cv2
import stereomideval.dataset
from metrics import mean_squared_err, birchfield_tomasi
import pt1
import pt2

# This file contained more functionality in other versions of the project, but I've stripped it down to upload it into the generic portfolio repo.

out_directory = "./output/"


def dict_to_csv(filename, mydict):
    # https://docs.python.org/3/library/csv.html
    # csv.writer boilerplate from https://stackoverflow.com/a/8685873 to store data
    # I would use pandas.Dataframe.to_csv() but I'm getting error messages attempting to add that to the conda env, probably a dependency conflict
    with open(filename, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])
        csv_file.close()


def runOnArbitrary(imL, imR, ground):
    part1 = pt1.NaiveApproach(imL, imR, window_size=3, max_delta=20)
    part1.run()

    dispL1, dispR1, av = part1.getDisparity()

    mean_squared_err_1 = mean_squared_err(dispL1, ground)
    bt_1 = birchfield_tomasi(dispL1, ground, normed=True, sumup=True)

    part2 = pt2.SGBMApproach(imL, imR, max_delta=100)
    part2.run(both=True)
    dispL2, dispR2 = part2.getFinalDisp()

    mean_squared_err_2 = mean_squared_err(dispL2, ground)
    bt_2 = birchfield_tomasi(dispL2, ground, normed=True, sumup=True)

    disp_img_name = "part1_arb.png"
    normalizedL1 = cv2.normalize(dispL1, None, 0, 255, cv2.NORM_MINMAX)
    normalizedL1 = normalizedL1.astype(np.uint8)
    cv2.imwrite(os.path.join(out_directory, disp_img_name), normalizedL1)

    disp_img_name = "part2_arb.png"
    normalizedL2 = cv2.normalize(dispL2, None, 0, 255, cv2.NORM_MINMAX)
    normalizedL2 = normalizedL2.astype(np.uint8)
    cv2.imwrite(os.path.join(out_directory, disp_img_name), normalizedL2)

    stats = {
        "Scene": ["Pt1", "Pt2"],
        "mean_squared_err with ground": [mean_squared_err_1, mean_squared_err_2],
        "BT Dissimilarity (ground)": [bt_1, bt_2],
    }
    dict_to_csv("arbitrary.csv", stats)


def run_arbitrary():
    # Alter this to whatever path
    imL_path = "imL.png"
    imR_path = "imR.png"
    # ground_path = "ground.png"
    ground_path = "ground.pfm"
    imL = cv2.imread(imL_path, 0)
    imR = cv2.imread(imR_path, 0)
    imL = np.copy(imL).astype(np.float32)
    imR = np.copy(imR).astype(np.float32)
    # if ground truth is stored in pfm comment this out otherwise uncomment this:
    # ground = cv2.imread(ground_path, 0)
    # ground= np.copy(ground)
    # if ground truth is stored in pfm uncomment this otherwise comment this:
    ground, scale = stereomideval.dataset.Dataset.load_pfm(ground_path)
    ground = ground.astype(np.uint8)
    runOnArbitrary(imL, imR, ground)


if __name__ == "__main__":
    run_arbitrary()
