import os, sys
import numpy as np
from PIL import Image
import cv2
import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import multiprocessing as mp
from multiprocessing import cpu_count

class Preprocess():
    # def __init__(self,path):
    #     self.path = path

    def facealine(self, zippar):
        
        rpath, wpath, size = zippar
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor, desiredFaceWidth=size)
        
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(rpath)
        try:
            image = imutils.resize(image, width=1200)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print(rpath)

        # show the original input image and detect faces in the grayscale
        # image
        try:
            rect = detector(gray, 2)[0]
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=size)
            faceAligned = fa.align(image, gray, rect)
            # display the output images
            cv2.imwrite(wpath, faceAligned)
        except:
            self.resizePic(rpath, wpath, size)

        return True

    def resizePic(self, rpath, wpath, size):
        # resize
        try:
            image = cv2.imread(rpath)
            image = imutils.resize(image, width=size)
            cv2.imwrite(wpath, image)
        except:
            print(rpath)

    #  multiprocessing
    def startPool(self, fun, rpath, wpath, size):
        print("start pool")
        size_list = [size for x in range(len(rpath))]   
        P = mp.Pool(processes=32)
        P.map(fun, zip(rpath, wpath, size_list))

        # get all path of image under dataset

    def getAllPath(self, path, writePath):
        ALL_picturePath = []
        ALL_writePath = []
        person_folderList = os.listdir(path)  # get all person's folder
        for person in person_folderList:  # get all picture under every person
            person_path = path + "/" + person
            picture_List = os.listdir(person_path)
            for picture in picture_List:  # resize every picture
                # creat new folder to write
                write_path = writePath + "/" + person + "/"
                if not os.path.exists(write_path):
                    os.makedirs(write_path)
                picture_path = person_path + "/" + picture
                if not os.path.exists(write_path + "/" + picture):
                    ALL_picturePath.append(picture_path)
                    ALL_writePath.append(write_path + "/" + picture)
            
        return ALL_picturePath, ALL_writePath

    # Youtube DB preprocess
    def mergeYoutube(self, youtubePath, writePath, size):
        person_folderList = os.listdir(youtubePath)  # get all person's folder
        for person in person_folderList:  # get all vedio under every person
            person_folder_path = youtubePath + "/" + person
            movie_folderList = os.listdir(person_folder_path)
            for movie in movie_folderList:  # get all picture under every vedio
                movie_path = person_folder_path + "/" + movie
                picture_List = os.listdir(movie_path)
                # get pre 100 picture
                if (len(picture_List) > 100):
                    picture_List = picture_List[0:101]
                # merge 100 picture
                img_ALL = np.zeros((size, size, 3))
                for picture in picture_List:
                    picture_path = movie_path + "/" + picture
                    img = Image.open(picture_path)  # open image
                    img_resize = img.resize((size, size), Image.BILINEAR)  # resize
                    img_array = np.array(img_resize)  # resize array
                    img_ALL += img_array  # immage add another

                img_ALL /= len(picture_List)  # get average
                merge_img = Image.fromarray(np.uint8(img_ALL))  # array change back to image

                # write file
                write_merge_path = writePath + "/" + person + "/"
                if not os.path.exists(write_merge_path):
                    os.makedirs(write_merge_path)
                merge_img.save(write_merge_path + movie + ".jpg")


# p = Preprocess()

# def alineDataset(rpath_DB, wpath_DB, size):
#     rpath, wpath = p.getAllPath(rpath_DB, wpath_DB)
#     p.startPool(p.facealine, rpath, wpath, size)

# vggReadPath = r"./dataset/vgg"
# vggWritePath = r"./dataset/vgg_224x224_aline_mp"

# alineDataset(vggReadPath, vggWritePath, 224)