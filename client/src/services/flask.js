import axios from "axios";
const flaskApp = axios.create({
  baseURL: "http://127.0.0.1:5000/",
});

export default flaskApp;
