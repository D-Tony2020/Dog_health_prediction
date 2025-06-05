import React, { useState } from 'react';
import { useEffect } from 'react';
import OutputInformation from '../OutputInformation/OutputInformation';
import SimpleChart from '../OutputInformation/SimpleChart';
import "./InputInformationOne.css";

const UserDataCard = () => {
  const [formData, setFormData] = useState({
    primaryBreed: 'American Pitbull Terrier',
    secondaryBreed: 'American Pitbull Terrier',
    weight: 'small',
    breedGroup: '',
    subBreed: '',
    breedType: '',
    breed_pure_or_mix: '1', // Default to 'pure'
    sex: '1', // Default to 'Female'
    ColdWheater_with_id: '1.0', // Default to 'cold_weather_1.0'
    physical_activity_hours_input: '',
    pa_surface: '1', // Default to 'concrete'
  });

  const [showPredictHealth, setShowPredictHealth] = useState(false);
  const [responseData, setResponseData] = useState(null);

  const handleInputChange = (prop) => (event) => {
    setFormData({ ...formData, [prop]: event.target.value });
  };

  const generateFeatureListAndSubmit = () => {
    // Create an array to hold the feature values
    const featureList = [];

 
   // Add value for age_diagnosis_yearsValue
   var age_diagnosis_yearsValue = document.getElementById("age_diagnosis_years").value;
   featureList.push(parseFloat(age_diagnosis_yearsValue));


    var primaryBreeds = document.querySelectorAll("#primaryBreed option");
    primaryBreeds.forEach(function (option, index, array) {
        // Check if it's not the last option
        if (index < array.length - 1) {
            featureList.push((option.value === document.getElementById("primaryBreed").value) ? 1 : 0);
        }
    });

     // Add values for od_annual_income_range_usd
     var selected_od_annual_income_range_usd = document.getElementById("od_annual_income_range_usd").value;
     featureList.push(parseFloat(selected_od_annual_income_range_usd));


      // Add values for education
      var selected_od_max_education = document.getElementById("od_max_education").value;
      featureList.push(parseFloat(selected_od_max_education));

      // Add values for pa_cold_weather_months_per_year
      var selected_pa_cold_weather_months_per_year = document.getElementById("pa_cold_weather_months_per_year").value;
      featureList.push(parseFloat(selected_pa_cold_weather_months_per_year));

      // Add values for pa_hot_weather_months_per_year
      var selected_pa_hot_weather_months_per_year = document.getElementById("pa_hot_weather_months_per_year").value;
      featureList.push(parseFloat(selected_pa_hot_weather_months_per_year));

      // Add values for oc_primary_census_division
      var oc_primary_census_division = document.querySelectorAll("#oc_primary_census_division option");
      oc_primary_census_division.forEach(function (option) {
          featureList.push((option.value === document.getElementById("oc_primary_census_division").value));
      });

      // Add values for pa_surface
      var pa_surface = document.querySelectorAll("#pa_surface option");
      pa_surface.forEach(function (option) {
          featureList.push((option.value === document.getElementById("pa_surface").value));
      });


      // Add value for physical_activity_hours
      var physicalActivityHoursValue = document.getElementById("physical_activity_hours_input").value;
      featureList.push(parseFloat(physicalActivityHoursValue));

      // Add value for pa_hot_weather_daily_hours_outside
      var pa_hot_weather_daily_hours_outside = document.getElementById("pa_hot_weather_daily_hours_outside").value;
      featureList.push(parseFloat(pa_hot_weather_daily_hours_outside));

      // Add value for pa_hot_weather_daily_hours_outside
      var pa_cold_weather_daily_hours_outside = document.getElementById("pa_cold_weather_daily_hours_outside").value;
      featureList.push(parseFloat(pa_cold_weather_daily_hours_outside));

      // Add value for pa_hot_weather_daily_hours_outside
      var pa_moderate_weather_daily_hours_outside = document.getElementById("pa_moderate_weather_daily_hours_outside").value;
      featureList.push(parseFloat(pa_moderate_weather_daily_hours_outside));

      // Add values for breed_pure_or_mix
      var breedPureOrMixValue = document.getElementById("breed_pure_or_mix").value;
      featureList.push((breedPureOrMixValue === "pure") ? 1 : 0);

      // Add values for df_feedings_per_day
      var df_feedings_per_day = document.getElementById("df_feedings_per_day").value;
      featureList.push((df_feedings_per_day === "1") ? 1 : 0);

      // Add values for df_diet_consistency
      var df_diet_consistency = document.getElementById("df_diet_consistency").value;
      featureList.push(parseFloat(df_diet_consistency));
  // Add values for df_daily_supplements
  var df_daily_supplements = document.getElementById("df_daily_supplements").value;
  featureList.push((df_daily_supplements === "1") ? 1 : 0);

  // Add values for df_primary_diet_component_organic
  var df_primary_diet_component_organic = document.getElementById("df_primary_diet_component_organic").value;
  featureList.push((df_primary_diet_component_organic === "1") ? 1 : 0);

  // Add values for df_primary_diet_component_grain_free
  var df_primary_diet_component_grain_free = document.getElementById("df_primary_diet_component_grain_free").value;
  featureList.push((df_primary_diet_component_grain_free === "1") ? 1 : 0);

  // Add values for dd_weight_lbs
  var dd_weight_lbs = document.getElementById("dd_weight_lbs").value;
  featureList.push(parseFloat(dd_weight_lbs));

  // Add values for rounded_age_years
  var rounded_age_years = document.getElementById("rounded_age_years").value;
  featureList.push(parseFloat(rounded_age_years));

  // Add values for sex
  var sex = document.querySelectorAll("#sex option");
  sex.forEach(function (option) {
      featureList.push((option.value === document.getElementById("sex").value));
  });

    console.log(featureList);

    // Set the featureList in the hidden input field
    const a = JSON.stringify(featureList);
    console.log(a);


    // Make an HTTP POST request to the Flask backend
    fetch('http://localhost:5000/predict_health_conditions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ featureList: featureList, formdata: formData }),
    })
      .then(response => response.json())
      .then(data => {
        // Handle the response from the backend
        console.log('Response from backend:', data);
        setResponseData(data.prediction);
      })
      .catch(error => {
        console.error('Error sending data to backend:', error);
        // Handle errors, if any
      });
    // Show the predicted health section
    setShowPredictHealth(true);
  };

  const handleSubmitForm = () => {
    // Simulate form submission (you can replace this with your actual API request)
    console.log('Form Data:', formData);
    generateFeatureListAndSubmit();
  };

  return (
    <div>
      <div className="m-16 min-h-screen flex flex-column items-center justify-center">
        <div id="input-field" className="bg-white p-6 rounded-lg shadow-md w-full lg:w-1/2">
          <h2 className="text-2xl mb-4 font-semibold text-center">Dog Information</h2>

          <label className='primary_label' for="breed_pure_or_mix">Breed Pure Or Mix</label>
            <select id="breed_pure_or_mix" 
            name="breed_pure_or_mix"
            value={formData.breed_pure_or_mix}
            onChange={handleInputChange('breed_pure_or_mix')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="pure">pure</option>
                <option value="mixed">mixed</option>

            </select>

          <label className='primary_label' for="weight">Weight</label>
            <select id="weight" 
            name="weight"
            value={formData.weight}
            onChange={handleInputChange('weight')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="small">0 - 20lbs</option>
                <option value="medium">21 - 60lbs</option>
                <option value="large">more than 61lbs</option>

            </select>

          <label className='primary_label' htmlFor="primaryBreed">Choose Primary Breed</label>
          <select
            id="primaryBreed"
            name="primaryBreed"
            value={formData.primaryBreed}
            onChange={handleInputChange('primaryBreed')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
             <option value="American Pitbull Terrier">American Pitbull Terrier</option>
                <option value="American Staffordshire Terrier">American Staffordshire Terrier</option>
                <option value="Australian Cattle Dog">Australian Cattle Dog</option>
                <option value="Australian Shepherd">Australian Shepherd</option>
                <option value="Basset Hound">Basset Hound</option>
                <option value="Beagle">Beagle</option>
                <option value="Bernese Mountain Dog">Bernese Mountain Dog</option>
                <option value="Bichon Frise">Bichon Frise</option>
                <option value="Border Collie">Border Collie</option>
                <option value="Boston Terrier">Boston Terrier</option>
                <option value="Boxer">Boxer</option>
                <option value="Bulldog">Bulldog</option>
                <option value="Cairn Terrier">Cairn Terrier</option>
                <option value="Catahoula Leopard Dog">Catahoula Leopard Dog</option>
                <option value="Cavalier King Charles Spaniel">Cavalier King Charles Spaniel</option>
                <option value="Chihuahua">Chihuahua</option>
                <option value="Chow Chow">Chow Chow</option>
                <option value="Cocker Spaniel">Cocker Spaniel</option>
                <option value="Collie">Collie</option>
                <option value="Dachshund">Dachshund</option>
                <option value="Doberman Pinscher">Doberman Pinscher</option>
                <option value="English Springer Spaniel">English Springer Spaniel</option>
                <option value="French Bulldog">French Bulldog</option>
                <option value="German Shepherd Dog">German Shepherd Dog</option>
                <option value="German Shorthaired Pointer">German Shorthaired Pointer</option>
                <option value="Golden Retriever">Golden Retriever</option>
                <option value="Great Dane">Great Dane</option>
                <option value="Great Pyrenees">Great Pyrenees</option>
                <option value="Greyhound">Greyhound</option>
                <option value="Havanese">Havanese</option>
                <option value="Jack Russell Terrier">Jack Russell Terrier</option>
                <option value="Labrador Retriever">Labrador Retriever</option>
                <option value="Maltese">Maltese</option>
                <option value="Miniature Pinscher">Miniature Pinscher</option>
                <option value="Miniature Schnauzer">Miniature Schnauzer</option>
                <option value="Newfoundland">Newfoundland</option>
                <option value="Other">Other</option>
                <option value="Pembroke Welsh Corgi">Pembroke Welsh Corgi</option>
                <option value="Pomeranian">Pomeranian</option>
                <option value="Poodle">Poodle</option>
                <option value="Poodle (Toy)">Poodle (Toy)</option>
                <option value="Pug">Pug</option>
                <option value="Rat Terrier">Rat Terrier</option>
                <option value="Rhodesian Ridgeback">Rhodesian Ridgeback</option>
                <option value="Rottweiler">Rottweiler</option>
                <option value="Shetland Sheepdog">Shetland Sheepdog</option>
                <option value="Shih Tzu">Shih Tzu</option>
                <option value="Siberian Husky">Siberian Husky</option>
                <option value="West Highland White Terrier">West Highland White Terrier</option>
                <option value="Yorkshire Terrier">Yorkshire Terrier</option>
                <option value="Unknown">Unknown</option>
          </select>
          <label className='primary_label' htmlFor="secondaryBreed">Choose Secondary Breed</label>
          <select
            id="secondaryBreed"
            name="secondaryBreed"
            value={formData.secondaryBreed}
            onChange={handleInputChange('secondaryBreed')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="American Pitbull Terrier">American Pitbull Terrier</option>
                <option value="American Staffordshire Terrier">American Staffordshire Terrier</option>
                <option value="Australian Cattle Dog">Australian Cattle Dog</option>
                <option value="Australian Shepherd">Australian Shepherd</option>
                <option value="Basset Hound">Basset Hound</option>
                <option value="Beagle">Beagle</option>
                <option value="Bernese Mountain Dog">Bernese Mountain Dog</option>
                <option value="Bichon Frise">Bichon Frise</option>
                <option value="Border Collie">Border Collie</option>
                <option value="Boston Terrier">Boston Terrier</option>
                <option value="Boxer">Boxer</option>
                <option value="Bulldog">Bulldog</option>
                <option value="Cairn Terrier">Cairn Terrier</option>
                <option value="Catahoula Leopard Dog">Catahoula Leopard Dog</option>
                <option value="Cavalier King Charles Spaniel">Cavalier King Charles Spaniel</option>
                <option value="Chihuahua">Chihuahua</option>
                <option value="Chow Chow">Chow Chow</option>
                <option value="Cocker Spaniel">Cocker Spaniel</option>
                <option value="Collie">Collie</option>
                <option value="Dachshund">Dachshund</option>
                <option value="Doberman Pinscher">Doberman Pinscher</option>
                <option value="English Springer Spaniel">English Springer Spaniel</option>
                <option value="French Bulldog">French Bulldog</option>
                <option value="German Shepherd Dog">German Shepherd Dog</option>
                <option value="German Shorthaired Pointer">German Shorthaired Pointer</option>
                <option value="Golden Retriever">Golden Retriever</option>
                <option value="Great Dane">Great Dane</option>
                <option value="Great Pyrenees">Great Pyrenees</option>
                <option value="Greyhound">Greyhound</option>
                <option value="Havanese">Havanese</option>
                <option value="Jack Russell Terrier">Jack Russell Terrier</option>
                <option value="Labrador Retriever">Labrador Retriever</option>
                <option value="Maltese">Maltese</option>
                <option value="Miniature Pinscher">Miniature Pinscher</option>
                <option value="Miniature Schnauzer">Miniature Schnauzer</option>
                <option value="Newfoundland">Newfoundland</option>
                <option value="Other">Other</option>
                <option value="Pembroke Welsh Corgi">Pembroke Welsh Corgi</option>
                <option value="Pomeranian">Pomeranian</option>
                <option value="Poodle">Poodle</option>
                <option value="Poodle (Toy)">Poodle (Toy)</option>
                <option value="Pug">Pug</option>
                <option value="Rat Terrier">Rat Terrier</option>
                <option value="Rhodesian Ridgeback">Rhodesian Ridgeback</option>
                <option value="Rottweiler">Rottweiler</option>
                <option value="Shetland Sheepdog">Shetland Sheepdog</option>
                <option value="Shih Tzu">Shih Tzu</option>
                <option value="Siberian Husky">Siberian Husky</option>
                <option value="West Highland White Terrier">West Highland White Terrier</option>
                <option value="Yorkshire Terrier">Yorkshire Terrier</option>
                <option value="Unknown">Unknown</option>
          </select>
          

        

          <label className='primary_label' htmlFor="sex">Sex</label>
          <select
            id="sex"
            name="sex"
            value={formData.sex}
            onChange={handleInputChange('sex')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1">Female</option>
            <option value="2">Male</option>
          </select>


          <label className='primary_label' htmlFor="ColdWheater_with_id">Cold Weather With ID</label>
          <select
            id="ColdWheater_with_id"
            name="ColdWheater_with_id"
            value={formData.ColdWheater_with_id}
            onChange={handleInputChange('ColdWheater_with_id')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1.0">cold_weather_1.0</option>
            <option value="2.0">cold_weather_2.0</option>
            <option value="3.0">cold_weather_3.0</option>
            <option value="4.0">cold_weather_4.0</option>
            <option value="5.0">cold_weather_5.0</option>
          </select>

         

          {/* Allow users to input physical activity hours */}
          <label className='primary_label' htmlFor="physical_activity_hours_input">Physical Activity Hours</label>
          <input
            type="number"
            id="physical_activity_hours_input"
            name="physical_activity_hours_input"
            min="0"
            value={formData.physical_activity_hours_input}
            onChange={handleInputChange('physical_activity_hours_input')}
            placeholder="Enter hours"
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          />
          <br />

          <label className='primary_label' htmlFor="pa_surface">Floor Type</label>
          <select
            id="pa_surface"
            name="pa_surface"
            value={formData.pa_surface}
            onChange={handleInputChange('pa_surface')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1">concrete</option>
            <option value="2">wood</option>
            <option value="3">other_hard</option>
            <option value="4">grass_dirt</option>
            <option value="5">gravel</option>
            <option value="6">sand</option>
            <option value="7">astroturf</option>
          </select>

          
          <label className='primary_label' for="pa_hot_weather_daily_hours_outside">Average Spent Activity Hours in Hot Weather</label>
            <select id="pa_hot_weather_daily_hours_outside" 
            name="pa_hot_weather_daily_hours_outside"
            value={formData.pa_hot_weather_daily_hours_outside}
            onChange={handleInputChange('pa_hot_weather_daily_hours_outside')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="1">Less than 3 hours</option>
                <option value="2">Between 3-8 hours</option>
                <option value="3">Between 8-16 hours</option>
                <option value="4">More than 16 hours</option>
                <option value="5">My dog does not spend time outdoors in hot weather</option>
            </select>
            

            <label className='primary_label' for="pa_cold_weather_daily_hours_outside">Average Spent Activity Hours in Cold Weather</label>
            <select id="pa_cold_weather_daily_hours_outside" 
            name="pa_cold_weather_daily_hours_outside"
            value={formData.pa_cold_weather_daily_hours_outside}
            onChange={handleInputChange('pa_cold_weather_daily_hours_outside')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="1">Less than 3 hours</option>
                <option value="2">Between 3-8 hours</option>
                <option value="3">Between 8-16 hours</option>
                <option value="4">More than 16 hours</option>
                <option value="5">My dog does not spend time outdoors in hot weather</option>
            </select>
            

            <label className='primary_label' for="pa_moderate_weather_daily_hours_outside">Average Spent Activity Hours in Moderate Weather</label>
            <select id="pa_moderate_weather_daily_hours_outside" 
            name="pa_moderate_weather_daily_hours_outside"
            value={formData.pa_moderate_weather_daily_hours_outside}
            onChange={handleInputChange('pa_moderate_weather_daily_hours_outside')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="1">Less than 3 hours</option>
                <option value="2">Between 3-8 hours</option>
                <option value="3">Between 8-16 hours</option>
                <option value="4">More than 16 hours</option>
                <option value="5">My dog does not spend time outdoors in hot weather</option>
            </select>
            

            <label className='primary_label' for="age_diagnosis_years">Age Diagnosis Years:</label>
            <input type="number" 
            id="age_diagnosis_years" 
            name="age_diagnosis_years" 
            min="0" 
            placeholder="Enter age_diagnosis_years"
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"/>
            

           <label className='primary_label' for="pa_cold_weather_months_per_year">Pat Cold Weather Months Per Year</label>
           <input type="number" 
           id="pa_cold_weather_months_per_year" 
           name="pa_cold_weather_months_per_year" 
           min="0" 
           max="12" 
           placeholder="Enter pa_cold_weather_months_per_year"
           className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"/>

         
           <label className='primary_label' for="pa_hot_weather_months_per_year">Pat Hot Weather Months Per Year</label>
           <input type="number" id="pa_hot_weather_months_per_year" name="pa_hot_weather_months_per_year" min="0" max="12" placeholder="Enter pa_hot_weather_months_per_year"  className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"/>
           

            <label className='primary_label' for="od_annual_income_range_usd">Annual Income Range</label>
            <select id="od_annual_income_range_usd" 
            name="od_annual_income_range_usd"
            value={formData.od_annual_income_range_usd}
            onChange={handleInputChange('od_annual_income_range_usd')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md">
                <option value="1">Less than $20,000</option>
                <option value="2">$20,000 - $39,999</option>
                <option value="3">$40,000 - $59,999</option>
                <option value="4">$60,000 - $79,999</option>
                <option value="5">$80,000 - $99,999</option>
                <option value="6">$100,000 - $119,999</option>
                <option value="7">$120,000 - $139,999</option>
                <option value="8">$140,000 - $159,999</option>
                <option value="9">$160,000 - $179,999</option>
                <option value="10">$180,000 or more</option>
                <option value="98">I’d prefer not to answer</option>
            </select>
            

            <label className='primary_label' for="oc_primary_census_division">Primary Census Division</label>
            <select id="oc_primary_census_division" 
            name="oc_primary_census_division"
            value={formData.oc_primary_census_division}
            onChange={handleInputChange('oc_primary_census_division')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md">
                <option value="1">oc_primary_census_division__1.0</option>
                <option value="2">oc_primary_census_division__2.0</option>
                <option value="3">oc_primary_census_division__3.0</option>
                <option value="4">oc_primary_census_division__4.0</option>
                <option value="5">oc_primary_census_division__5.0</option>
                <option value="6">oc_primary_census_division__6.0</option>
                <option value="7">oc_primary_census_division__7.0</option>
                <option value="8">oc_primary_census_division__8.0</option>
                <option value="9">oc_primary_census_division__9.0</option>
            </select>
          


            <label className='primary_label' for="od_max_education">Owner's Education level</label>
            <select id="od_max_education" 
            name="od_max_education"
            value={formData.od_max_education}
            onChange={handleInputChange('od_max_education')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md">
                <option value="0">No schooling completed</option>
                <option value="1">Nursery school to 8th grade</option>
                <option value="2">Some high school, no diploma</option>
                <option value="3">High school graduate</option>
                <option value="4">High school equivalent degree (such as GED)</option>
                <option value="5">Some college credit, no degree</option>
                <option value="6">Trade, technical, or vocational training</option>
                <option value="7">Associate degree</option>
                <option value="8">Bachelor’s degree</option>
                <option value="9">Master’s degree</option>
                <option value="10">Professional degree (such as DVM, MD, DDS, or JD)</option>
                <option value="11">Doctorate degree (such as PhD, DrPH, or DPhil)</option>
            </select>
           

            <label className='primary_label' for="df_feedings_per_day">Feeding Frequency</label>
            <select id="df_feedings_per_day" 
            name="df_feedings_per_day"
            value={formData.df_feedings_per_day}
            onChange={handleInputChange('df_feedings_per_day')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md">
                <option value="1">Once</option>
                <option value="2">Twice</option>
                <option value="3">Three or more</option>
                <option value="4">Free fed (filling up bowl when empty or always having food available)</option>
            </select>
            

           
            <label className='primary_label' for="df_diet_consistency">Consistency Of Dog’s Daily Diet</label>
            <select id="df_diet_consistency" 
            name="df_diet_consistency"
            value={formData.df_diet_consistency}
            onChange={handleInputChange('df_diet_consistency')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="1">Very consistent</option>
                <option value="2">Somewhat consistent</option>
                <option value="3">Not at all consistent</option>
                <option value="98">Other</option>
            </select>
            

          
            <label className='primary_label' for="df_daily_supplements">Supplements One Or More Times A Day</label>
            <select id="df_daily_supplements" 
            name="df_daily_supplements"
            value={formData.df_daily_supplements}
            onChange={handleInputChange('df_daily_supplements')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md">
                <option value="1">Yes</option>
                <option value="2">No</option>
            </select>
            

      
            <label className='primary_label' for="df_primary_diet_component_organic">The Primary Component Of your Dog’s Diet Is Organic</label>
            <select id="df_primary_diet_component_organic" 
            name="df_primary_diet_component_organic"
            value={formData.df_primary_diet_component_organic}
            onChange={handleInputChange('df_primary_diet_component_organic')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md">
                <option value="1">Yes</option>
                <option value="2">No</option>
            </select>
           

    
            <label className='primary_label' for="df_primary_diet_component_grain_free">The Primary Component Of your Dog’s Diet Is Grain Free</label>
            <select id="df_primary_diet_component_grain_free" 
            name="df_primary_diet_component_grain_free"
            value={formData.df_primary_diet_component_grain_free}
            onChange={handleInputChange('df_primary_diet_component_grain_free')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
            >
                <option value="1">Yes</option>
                <option value="2">No</option>
            </select>
            

           
            <label className='primary_label' for="dd_weight_lbs">Dog Weight(lbs)</label>
            <input type="number" id="dd_weight_lbs" name="dd_weight_lbs" min="0" placeholder="Enter dd_weight_lbs" className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"/>
            

            
            <label className='primary_label' for="rounded_age_years">Rounded Age years</label>
            <input type="number" id="rounded_age_years" name="rounded_age_years" min="0" placeholder="Enter rounded_age_years" className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"/>
          


          {/* Hidden input field for storing selection results */}
          <input type="hidden" id="primaryBreedValue" name="primaryBreedValue" value="0" />
          <input type="hidden" id="subBreedValue" name="subBreedValue" value="false" />
          <input type="hidden" id="breedGroupValue" name="breedGroupValue" value="false" />
          <input type="hidden" id="breedTypeValue" name="breedTypeValue" value="false" />
          <input type="hidden" id="breed_pure_or_mixValue" name="breed_pure_or_mixValue" value="0" />
          <input type="hidden" id="sexValue" name="sexValue" value="false" />
          <input type="hidden" id="ColdWheater_with_idValue" name="ColdWheater_with_idValue" value="0" />
          <input type="hidden" id="physical_activity_hours_value" name="physical_activity_hours_value" value="0" />
          <input type="hidden" id="pa_surfaceValue" name="pa_surfaceValue" value="false" />

          {/* Hidden input field to store the feature list */}
          <input type="hidden" id="featureList" name="featureList" value="" />

          <br />

          {/* When users click submit */}
          <button
            className="bg-blue-500 text-white px-6 py-3 rounded-md hover:bg-blue-600 transition duration-300 block mx-auto"
            onClick={handleSubmitForm}
          >
            Submit
          </button>

        </div>

      </div>
      <div className="m-16 min-h-screen flex flex-column items-center justify-center">
        <div className="bg-white p-6 rounded-lg shadow-md w-full lg:w-1/2">
          {responseData==null? <div><h1>No Information to show.</h1></div>:
            <OutputInformation responseData={responseData}></OutputInformation>
            // <SimpleChart></SimpleChart>
          }
        </div>
      </div>
    </div>
  );
};

export default UserDataCard;



