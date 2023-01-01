<template>
    <va-navbar color="primary">
        <template #left>
            <va-navbar-item>
                <va-icon name="coronavirus" class="mr-2" />
                <span class="header-text">Predicting COVID-19 virus using Transfer Learning and Fine tuning.</span>
            </va-navbar-item>
        </template>
        <template #right>
            <va-button to="/" :rounded="false" class="mr-4" color="#FFFFFF" text-color="primary">Home</va-button>
            <va-button to="/predict" :rounded="false" class="mr-4" color="#FFFFFF" text-color="primary">Make a Prediction</va-button>
        </template>
    </va-navbar>
</template>

<script setup>

</script>

<style>
.header-text {
    font-weight: bolder;
    font-size: 1.1rem;
}
</style>