


class Metrics:
    def __init__(self, test_regularization):
        self.baseline = test_regularization['Baseline']
        self.l1 = test_regularization['L1']
        self.l2 = test_regularization['L2']
        self.dropout = test_regularization['Dropout']

    def get_metrics(self, param):
        if param == "val_AUC":
            return self.__get_AUC()
        
        elif param == "val_Accuracy":
            return self.__get_Accuracy()

        elif param == "val_F1_score":
            return self.__get_F1_score()

        elif param == "val_Precision":
            return self.__get_Precision()

        elif param == "val_Recall":
            return self.__get_Recall()
        
        elif param == "val_loss":
            return self.__get_loss()

        else:
            return self.__get_learning_rate()


    def __get_AUC(self):
        return [
            self.baseline["val_AUC"],
            self.l1["val_AUC"],
            self.l2["val_AUC"],
            self.dropout["val_AUC"]
            ]
    
    def __get_Accuracy(self):
        return [
            self.baseline["val_Accuracy"],
            self.l1["val_Accuracy"],
            self.l2["val_Accuracy"],
            self.dropout["val_Accuracy"]
            ]
    
    def __get_F1_score(self):
        return [
            self.baseline["val_F1_score"],
            self.l1["val_F1_score"],
            self.l2["val_F1_score"],
            self.dropout["val_F1_score"]
            ]
    
    def __get_Precision(self):
        return [
            self.baseline["val_Precision"],
            self.l1["val_Precision"],
            self.l2["val_Precision"],
            self.dropout["val_Precision"]
            ]
    
    def __get_Recall(self):
        return [
            self.baseline["val_Recall"],
            self.l1["val_Recall"],
            self.l2["val_Recall"],
            self.dropout["val_Recall"]
            ]
    
    def __get_loss(self):
        return [
            self.baseline["val_loss"],
            self.l1["val_loss"],
            self.l2["val_loss"],
            self.dropout["val_loss"]
            ]
    
    def __get_learning_rate(self):
        return [
            self.baseline["val_learning_rate"],
            self.l1["val_learning_rate"],
            self.l2["val_learning_rate"],
            self.dropout["val_learning_rate"]
            ]


