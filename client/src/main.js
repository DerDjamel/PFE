import { createApp } from 'vue';
import App from './App.vue';
import Router from "./routes";
import { createVuestic } from 'vuestic-ui'; 
import 'vuestic-ui/dist/vuestic-ui.css'; 
import "./assets/css/overrides.css";




const app = createApp(App);
app.use(Router);
app.use(createVuestic());
app.mount('#app');
