import torch

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001
    def __init__(self,
               temperature=100,
               cos_distance='multiplication'):

        self.temperature = temperature
        self.cos_distance = cos_distance

    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)

    @staticmethod
    def pairwise_cos_distance(A, B):

        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        prod = torch.matmul(normalized_A, normalized_B.transpose(-2, -1).conj())
        return 1 - prod

    @staticmethod
    def pairwise_multiplication_similarity(A, B):
        """Pairwise multiplication similarity between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise multiplication similarity between A and B.
        """
        return torch.matmul(A, B.transpose(-2, -1).conj())

    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance=='multiplication':
            distance_matrix = SNNLCrossEntropy.pairwise_multiplication_similarity(A, B)
        elif cos_distance=='euclidean':
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
        elif cos_distance=='cosine':
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            raise ValueError('cos_distance should be one of "multiplication", "euclidean", "cosine"')

        STABILITY_EPS = 0.00001
        return torch.exp(-(distance_matrix / (temp + STABILITY_EPS)))

    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0]).to(x.device)
        return f / (f.sum(axis=1).unsqueeze(1) + SNNLCrossEntropy.STABILITY_EPS)

    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)

    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
                                    SNNLCrossEntropy.same_label_mask(y, y)

    @staticmethod
    def SNNL(x, y, temp=100, cos_distance='multiplication'):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        STABILITY_EPS = 0.00001
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(summed_masked_pick_prob + STABILITY_EPS).mean()



