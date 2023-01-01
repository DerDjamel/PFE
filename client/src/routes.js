import { createRouter, createWebHistory } from "vue-router"

import Home from "./pages/HomePage.vue"
import PredicitonPage from "./pages/PredictionPage.vue"



const routes = [
  { path: "/", component: Home },
  { path: "/predict", component: PredicitonPage },
];

const router = createRouter({
  history: createWebHistory(),
  routes, 
});


export default router;

