import flaskApp from "./flask";

export default {
  routes: {
    classification: "classification/predict",
    clustering: "clustering/predict",
  },
  classification(data) {
    return flaskApp.post(this.routes.classification, data);
  },

  clustering(data) {
    return flaskApp.post(this.routes.clustering, data);
  },
};
