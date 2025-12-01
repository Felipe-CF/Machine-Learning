


class ModelMetrics:
    def __init__(self, history):
        self.__auc = (history['AUC'], history['val_AUC'])
        self.__acc = (history['Accuracy'], history['val_Accuracy'])
        self.__f1 = (history['F1_score'], history['val_F1_score'])
        self.__prec = (history['Precision'], history['val_Precision'])
        self.__rec = (history['Recall'], history['val_Recall'])
        self.__loss = (history['loss'],history['val_loss'])
        self.__lr = history['learning_rate']

    def get_metrics(self, param=None):
        if param == "AUC":
            return self.__get_AUC()
        
        elif param == "Accuracy":
            return self.__get_Accuracy()

        elif param == "F1_score":
            return self.__get_F1_score()

        elif param == "Precision":
            return self.__get_Precision()

        elif param == "Recall":
            return self.__get_Recall()
        
        elif param == "loss":
            return self.__get_loss()
        
        elif param == "lr":
            return self.__get_learning_rate()

        else:
            return self.__get_all()


    def __get_AUC(self):
        return self.__auc
    
    def __get_Accuracy(self):
        return self.__acc
    
    def __get_F1_score(self):
        return self.__f1
    
    def __get_Precision(self):
        return self.__prec
    
    def __get_Recall(self):
        return self.__rec
    
    def __get_loss(self):
        return self.__loss
    
    def __get_learning_rate(self):
        return self.__lr
    
    def __get_all(self):
        return [self.__auc, self.__acc, self.__f1, self.__prec, self.__rec, self.__loss]


