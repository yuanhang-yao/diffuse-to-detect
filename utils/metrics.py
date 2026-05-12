import numpy as np
from skimage import measure

class SigmoidMetric():
    def __init__(self, score_thresh=0.5):
        self.score_thresh = score_thresh
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        predict = (output > self.score_thresh).astype('int64')
        target = (target > self.score_thresh).astype('int64')

        pixel_labeled = np.sum(target > 0)
        pixel_correct = np.sum((predict == target) * (target > 0))
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1
        nbins = 1
        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')
        target = (target.cpu().detach().numpy() > self.score_thresh).astype('int64')

        intersection = predict * (predict == target)


        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union

class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        inter_arr, union_arr = self.batch_intersection_union(preds, labels)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target):


        mini = 1
        maxi = 1
        nbins = 1

        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')
        target = (target.cpu().detach().numpy() > self.score_thresh).astype('int64')

        intersection = predict * (predict == target)

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):

            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr

class PD_FA_2():
    def __init__(self, nclass):
        super(PD_FA_2, self).__init__()
        self.nclass = nclass
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target = 0
        self.all_pixel = 0

    def update(self, preds, labels):

        predits = (preds > 0.5).cpu().detach().numpy().astype('int64')
        labelss = (labels > 0.5).cpu().detach().numpy().astype('int64')

        batch_size = predits.shape[0]
        for b in range(batch_size):
            pred_single = predits[b, 0]
            label_single = labelss[b, 0]
            h, w = pred_single.shape
            self.all_pixel += h * w

            image = measure.label(pred_single, connectivity=2)
            coord_image = measure.regionprops(image)

            label = measure.label(label_single, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target += len(coord_label)

            image_area_total = []
            image_area_match = []
            distance_match = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                image_area_total.append(area_image)

            coord_image_copy = coord_image.copy()
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image_copy)):
                    if m >= len(coord_image_copy):
                        break
                    centroid_image = np.array(list(coord_image_copy[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image_copy[m].area)
                    if distance < 3:
                        distance_match.append(distance)
                        image_area_match.append(area_image)

                        del coord_image_copy[m]
                        break

            dismatch = np.sum(image_area_total) - np.sum(image_area_match)
            self.FA += dismatch
            self.PD += len(distance_match)

    def get(self):
        Final_FA = self.FA / self.all_pixel if self.all_pixel > 0 else 0
        Final_PD = self.PD / self.target if self.target > 0 else 0

        return Final_FA,Final_PD

    def reset(self):
        self.FA  = 0
        self.PD  = 0
        self.target = 0
        self.all_pixel = 0
