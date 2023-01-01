<script setup>
import NavigationBar from "./components/NavigationBar.vue";
import { ref } from "vue"
let image = ref([]);
</script>

<template>
  <NavigationBar></NavigationBar>
  <router-view></router-view>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,400;1,700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
</style>
