import numpy as np
import torch


class ClassificationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.true_positives = [0] * num_classes
        self.false_positives = [0] * num_classes
        self.true_negatives = [0] * num_classes
        self.false_negatives = [0] * num_classes

        self.predicted_probs = [None for _ in range(num_classes)]
        self.ground_truths = None

    def update(self, predicted_probs, ground_truths):
        predictions = predicted_probs.argmax(dim=1)
        for i in range(self.num_classes):
            bitmap_1 = (predictions == i)
            bitmap_2 = (ground_truths == i)
            bitmap_3 = bitmap_1 ^ bitmap_2

            tp = (bitmap_1 & bitmap_2).sum().item()
            fp = (bitmap_3 & bitmap_1).sum().item()
            fn = (bitmap_3 & bitmap_2).sum().item()
            tn = len(predictions) - tp - fp - fn

            self.true_positives[i] += tp
            self.false_positives[i] += fp
            self.false_negatives[i] += fn
            self.true_negatives[i] += tn

            probs = predicted_probs[:, i]
            self.predicted_probs[i] = probs if self.predicted_probs[i] is None \
                else torch.cat([self.predicted_probs[i], probs])

        self.ground_truths = ground_truths if self.ground_truths is None \
            else torch.cat([self.ground_truths, ground_truths])

    @property
    def precisions(self):
        precisions = []
        for i in range(self.num_classes):
            tp = self.true_positives[i]
            fp = self.false_positives[i]
            precisions.append(np.divide(tp, tp + fp))

        return precisions

    @property
    def recalls(self):
        recalls = []
        for i in range(self.num_classes):
            tp = self.true_positives[i]
            fn = self.false_negatives[i]
            recalls.append(np.divide(tp, tp + fn))

        return recalls

    @property
    def sensitivities(self):
        return self.recalls

    @property
    def specificities(self):
        specificities = []
        for i in range(self.num_classes):
            tn = self.true_negatives[i]
            fp = self.false_positives[i]
            specificities.append(np.divide(tn, tn + fp))

        return specificities

    @property
    def macro_precision(self):
        return sum(self.precisions) / self.num_classes

    @property
    def macro_recall(self):
        return sum(self.recalls) / self.num_classes

    @property
    def micro_precision(self):
        tp = sum(self.true_positives)
        fp = sum(self.false_positives)

        return np.divide(tp, tp + fp)

    @property
    def micro_recall(self):
        tp = sum(self.true_positives)
        fn = sum(self.false_negatives)

        return np.divide(tp, tp + fn)

    @property
    def f1_scores(self):
        f1_scores = []
        precisions = self.precisions
        recalls = self.recalls
        for i in range(self.num_classes):
            f1_scores.append(np.divide(2 * precisions[i] * recalls[i], precisions[i] + recalls[i]))

        return f1_scores

    @property
    def macro_f1(self):
        return sum(self.f1_scores) / self.num_classes

    @property
    def micro_f1(self):
        micro_precision = self.micro_precision
        micro_recall = self.micro_recall

        return np.divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    @property
    def accuracies(self):
        accuracies = []
        for i in range(self.num_classes):
            tp = self.true_positives[i]
            fp = self.false_positives[i]
            tn = self.true_negatives[i]
            fn = self.false_negatives[i]
            accuracies.append(np.divide(tp + tn, tp + fp + tn + fn))

        return accuracies

    @property
    def average_accuracy(self):
        return sum(self.accuracies) / self.num_classes

    @property
    def overall_accuracy(self):  # top-1 accuracy
        num_samples = len(self.ground_truths)

        return sum(self.true_positives) / num_samples

    @property
    def roc_auc_scores(self):
        tprs = [[] for _ in range(self.num_classes)]
        fprs = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):  # one-vs-all
            probs = self.predicted_probs[i]
            if probs is None:
                return

            bitmap_1 = (self.ground_truths == i)
            for threshold in np.linspace(0, 1, num=200):
                bitmap_2 = (probs >= threshold)
                bitmap_3 = bitmap_2 ^ bitmap_1

                tp = (bitmap_2 & bitmap_1).sum().item()
                fp = (bitmap_3 & bitmap_2).sum().item()
                fn = (bitmap_3 & bitmap_1).sum().item()
                tn = len(probs) - tp - fp - fn

                tprs[i].insert(0, np.divide(tp, tp + fn))
                fprs[i].insert(0, np.divide(fp, fp + tn))

        auc_scores = [np.trapz(tprs[i], fprs[i]) for i in range(self.num_classes)]

        return auc_scores

    @property
    def macro_roc_auc(self):
        return sum(self.roc_auc_scores) / self.num_classes

    @property
    def pr_auc_scores(self):
        precisions = [[] for _ in range(self.num_classes)]
        recalls = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):  # one-vs-all
            probs = self.predicted_probs[i]
            if probs is None:
                return

            bitmap_1 = (self.ground_truths == i)
            for threshold in np.linspace(0, 1, num=200):
                bitmap_2 = (probs >= threshold)
                bitmap_3 = bitmap_2 ^ bitmap_1

                tp = (bitmap_2 & bitmap_1).sum().item()
                if tp == 0:  # precision is not defined
                    continue

                fp = (bitmap_3 & bitmap_2).sum().item()
                fn = (bitmap_3 & bitmap_1).sum().item()

                precisions[i].insert(0, np.divide(tp, tp + fp))
                recalls[i].insert(0, np.divide(tp, tp + fn))

        auc_scores = [np.trapz(precisions[i], recalls[i]) for i in range(self.num_classes)]

        return auc_scores

    @property
    def macro_pr_auc(self):
        return sum(self.pr_auc_scores) / self.num_classes


class SegmentationMetrics:  # image-level evaluation for dense predictions
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.per_image_f1_scores = [[] for _ in range(num_classes)]
        self.per_image_jaccard_scores = [[] for _ in range(num_classes)]

    def update(self, predictions, ground_truths):
        for prediction, ground_truth in zip(predictions, ground_truths):
            for i in range(self.num_classes):
                bitmap_1 = (prediction == i)
                bitmap_2 = (ground_truth == i)

                intersections = (bitmap_1 & bitmap_2).sum().item()
                total = bitmap_1.sum().item() + bitmap_2.sum().item()
                unions = total - intersections

                self.per_image_f1_scores[i].append(np.divide(2 * intersections, total))
                self.per_image_jaccard_scores[i].append(np.divide(intersections, unions))

    @property
    def f1_scores(self):
        return [np.mean(scores) for scores in self.per_image_f1_scores]

    def jaccard_scores(self):
        return [np.mean(scores) for scores in self.per_image_jaccard_scores]
