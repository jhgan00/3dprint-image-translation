import torch
import numpy as np


def cri(img, max_rad):
    """ 개별 이미지의 CRI 값을 계산
    :param img: np.ndarray 또는 torch.Tensor (H, W), 그레이스케일 이미지
    :param max_rad:
    :return:
    """

    img_row = []  # row 길이 측정
    for i in range(img.shape[0]):
        img_row.append(img[i])

    img_col = []  # col 길이 측정
    for i in range(img.shape[1]):
        img_col.append(img[:, i])

    # bound는 왼쪽에서부터 배경의 영역을 지정
    a = len(img_row)
    b = len(img_col)

    bound = int(b / 120)

    # 영역 나눠서 평균값 추출
    back_avg = np.average(img[0:a, 0:bound])
    target_avg = np.average(img[0:a, bound:b])

    # 평균 픽셀값을 rad값으로 변환
    back_rad = back_avg * max_rad / 255
    target_rad = target_avg * max_rad / 255
    cri = abs(target_rad - back_rad)

    return cri


def mean_absolute_cri_error(pred, true, max_rad=100, return_error_array=False):
    """
    :param pred: torch.Tensor[torch.uint8] (N x H x W)
    :param true: torch.Tensor[torch.uint8] (N x H x W)
    :param max_rad: LWIR 의 경우 100, MWIR 의 경우 20
    :return: 평균 CRI 에러율 (return_error_array = True 인 경우 에러 배열, 평균 에러의 튜플을 반환)
    """
    assert pred.shape == true.shape
    pred_cri = np.array([cri(p, max_rad=max_rad) for p in pred])
    true_cri = np.array([cri(t, max_rad=max_rad) for t in true])
    abs_error = abs(pred_cri - true_cri) / true_cri * 100
    if return_error_array:
        return abs_error.mean(), abs_error
    else:
        return abs_error.mean()


def mean_pixel_loss(pred, true, device='cpu'):
    """
    :param pred: torch.Tensor (N x H x W)
    :param true: torch.Tensor (N x H x W)
    :return: 픽셀 로스
    """
    assert pred.shape == true.shape
    pred = pred.to(device)
    true = true.to(device)

    return (pred - true).abs().sum(dim=list(range(1, len(pred.shape)))).mean()


class ScoreMetric:

    def __init__(self):
        super(ScoreMetric, self).__init__()

    def cal(self, real_error, fake_B, pixel=255):
        batch = fake_B.shape[0]  # batch size
        neg_list = []  # negative error list, batch size length
        pos_list = []  # positive error list, batch size length
        error_list = []  # error list, batch size length
        error_rate_list = []  # error rate list, batch size length

        for i in range(batch):
            fake = fake_B[i].cpu()
            neg_num = (fake < 0).sum()  # number of negative value
            pos_num = (fake > 0).sum()  # number of postive value

            if isinstance(fake, np.ndarray):  # cv2와 ndarray일 때
                neg = np.where(fake > 0, 0, fake).sum() / (neg_num * pixel)  # negative error
                pos = np.where(fake < 0, 0, fake).sum() / (pos_num * pixel)  # positive error

            elif isinstance(fake, torch.Tensor):  # tensor 일 때
                neg = np.where(fake > 0, 0, fake).sum() / neg_num  # negative error
                pos = np.where(fake < 0, 0, fake).sum() / pos_num  # positive error

            else:
                raise TypeError

            error = (abs(neg) + pos) / 2  # average error
            error_rate = abs(real_error[i] - error) / real_error[i]  # error rate
            neg_list.append(neg)
            pos_list.append(pos)
            error_list.append(error)
            error_rate_list.append(error_rate)

        return neg_list, pos_list, error_list, error_rate_list


if __name__ == "__main__":
    sm = ScoreMetric()

    x = torch.normal(0, 1, (1, 224, 224))
    y = torch.normal(0, 1, (1, 224, 224))

    sm.cal(x, y)

