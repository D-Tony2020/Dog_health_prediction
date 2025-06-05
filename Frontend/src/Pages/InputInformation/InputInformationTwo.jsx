import React, { useState } from 'react';
import { useEffect } from 'react';
import OutputInformation from '../OutputInformation/OutputInformation';
import SimpleChart from '../OutputInformation/SimpleChart';
import "./InputInformationOne.css";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import PieChart from '../OutputInformation/StylishBarChart';
import StylishBarChart from '../OutputInformation/StylishBarChart';


const UserDataCard = () => {
    const [formData, setFormData] = useState({
        primaryBreed: 'American Pitbull Terrier',
        secondaryBreed: 'American Pitbull Terrier',
        weight: '',
        state: 'Maine',
        breedGroup: '',
        subBreed: '',
        breedType: '',
        breed_pure_or_mix: 'pure', // Default to 'pure'
        sex: '1', // Default to 'Male'
        ColdWheater_with_id: '1.0', // Default to 'cold_weather_1.0'
        physical_activity_hours: '',
        pa_surface: '1', // Default to 'concrete'
        rounded_age_years: '',
    });

    const [showPredictHealth, setShowPredictHealth] = useState(false);
    const [responseData, setResponseData] = useState(null);
    const dogAge = [];

    const handleInputChange = (prop) => (event) => {
        setFormData({ ...formData, [prop]: event.target.value });
    };

    const generateFeatureListAndSubmit = () => {
        // Create an array to hold the feature values
        const featureList = [];



    const unknownPrimaryBreed = formData.primaryBreed == 'Unknown';
    const unknownSecondaryBreed = formData.secondaryBreed != 'Unknown';
    // console.log(unknownPrimaryBreed, unknownSecondaryBreed)

    const data = [0.7114651843137689, 0.703769881719169, 0.7023164279123049, 0.6313146170861812, 
        0.7402321561336324, 0.7640955968531346, 0.6557358731531935, 0.7108303529643931, 
        0.7257548046331452, 0.7980000344784488, 0.6634657632521148, 0.6643505796408098, 
        0.7119546231287156, 0.7045672560794011, 0.8364834828593533, 0.7696097581778122, 
        0.6513466583661263, 0.6364062783489701, 0.5930778686116873, 0.7711782579152425];
      
    
    const conditionTypes = [
        'Eye', 'Ear/Nose/Throat', 'Mouth/Dental/Oral', 'Skin', 'Cardiac', 'Respiratory',
        'Gastrointestinal', 'Liver/Pancreas', 'Kidney/Urinary', 'Reproductive', 'Bone/Orthopedic',
        'Brain/Neurologic', 'Endocrine', 'Hematopoietic', 'Other Congenital Disorder',
        'Infection/Parasites', 'Toxin Consumption', 'Trauma', 'Immune-mediated', 'cancer'
      ];

    if (unknownPrimaryBreed && unknownSecondaryBreed) {
      // Show toast notification for unknown primary breed
      toast.warning("Please choose the known Secondary breed in the Primary breed field");
      return;  // Do not proceed with the submission
    }

        var primaryBreeds = document.querySelectorAll("#primaryBreed option");
        primaryBreeds.forEach(function (option, index, array) {
            // Check if it's not the last option
            if (index < array.length - 1) {
                featureList.push((option.value === document.getElementById("primaryBreed").value) ? 1 : 0);
            }
        });

        // Add values for breed_pure_or_mix
        var breedPureOrMixValue = document.getElementById("breed_pure_or_mix").value;
        featureList.push((breedPureOrMixValue === "pure") ? 1 : 0);

        // Add values for rounded_age_years
        var rounded_age_years = document.getElementById("rounded_age_years").value;
        featureList.push(parseFloat(rounded_age_years));

        // Add values for sex
        var sex = document.querySelectorAll("#sex option");
        sex.forEach(function (option) {
            featureList.push((option.value === document.getElementById("sex").value));
        });

        // Add values for dd_weight_lbs
        var dd_weight_lbs = document.getElementById("dd_weight_lbs").value;
        console.log(parseFloat(dd_weight_lbs));
        featureList.push(parseFloat(dd_weight_lbs));

        // Add values for pa_cold_weather_months_per_year
        var selected_pa_cold_weather_months_per_year = document.getElementById("pa_cold_weather_months_per_year").value;
        featureList.push(parseFloat(selected_pa_cold_weather_months_per_year));

        // Add values for pa_hot_weather_months_per_year
        var selected_pa_hot_weather_months_per_year = document.getElementById("pa_hot_weather_months_per_year").value;
        featureList.push(parseFloat(selected_pa_hot_weather_months_per_year));

        // Add values for pa_surface
        var pa_surface = document.querySelectorAll("#pa_surface option");
        pa_surface.forEach(function (option) {
            featureList.push((option.value === document.getElementById("pa_surface").value));
        });

        // Add value for physical_activity_hours
        var physicalActivityHoursValue = document.getElementById("physical_activity_hours").value;
        featureList.push(parseFloat(physicalActivityHoursValue));

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
        console.log(featureList);

        // Set the featureList in the hidden input field
        const a = JSON.stringify(featureList);
        // console.log(a);


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
                // console.log('Response from backend:', data);
                // console.log(JSON.stringify(data));
                setResponseData(data.prediction);
            })
            .catch(error => {
                console.error('Error sending data to backend:', error);
                toast.warning("Please fill the input fields correctly");
                // Handle errors, if any
            });
        // Show the predicted health section
        setShowPredictHealth(true);
    };

    const handleSubmitForm = () => {
        // Simulate form submission (you can replace this with your actual API request)
        // console.log('Form Data:', formData);
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
                    
                    {formData.breed_pure_or_mix === 'mixed' && (
    <>
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
</>
)}
                    <ToastContainer />




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

                    <label for="state">Please select your state：</label>
                    <select id="state"
                        name="state"
                        value={formData.state}
                        onChange={handleInputChange('state')}
                        className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
                    >
                        <optgroup label="New England Division">
                            <option value="Maine">Maine</option>
                            <option value="NewHampshire">New Hampshire</option>
                            <option value="Vermont">Vermont</option>
                            <option value="Massachusetts">Massachusetts</option>
                            <option value="RhodeIsland">Rhode Island</option>
                            <option value="Connecticut">Connecticut</option>
                        </optgroup>
                        <optgroup label="Middle Atlantic Division">
                            <option value="NewYork">New York</option>
                            <option value="Pennsylvania">Pennsylvania</option>
                            <option value="NewJersey">New Jersey</option>
                        </optgroup>
                        <optgroup label="East North Central Division">
                            <option value="Ohio">Ohio</option>
                            <option value="Indiana">Indiana</option>
                            <option value="Illinois">Illinois</option>
                            <option value="Michigan">Michigan</option>
                            <option value="Wisconsin">Wisconsin</option>
                        </optgroup>
                        <optgroup label="West North Central Division">
                            <option value="Missouri">Missouri</option>
                            <option value="NorthDakota">North Dakota</option>
                            <option value="SouthDakota">South Dakota</option>
                            <option value="Nebraska">Nebraska</option>
                            <option value="Kansas">Kansas</option>
                            <option value="Minnesota">Minnesota</option>
                            <option value="Iowa">Iowa</option>
                        </optgroup>
                        <optgroup label="South Atlantic Division">
                            <option value="Delaware">Delaware</option>
                            <option value="Maryland">Maryland</option>
                            <option value="DistrictOfColumbia">District of Columbia</option>
                            <option value="Virginia">Virginia</option>
                            <option value="WestVirginia">West Virginia</option>
                            <option value="NorthCarolina">North Carolina</option>
                            <option value="SouthCarolina">South Carolina</option>
                            <option value="Georgia">Georgia</option>
                            <option value="Florida">Florida</option>
                        </optgroup>
                        <optgroup label="East South Central Division">
                            <option value="Kentucky">Kentucky</option>
                            <option value="Tennessee">Tennessee</option>
                            <option value="Mississippi">Mississippi</option>
                            <option value="Alabama">Alabama</option>
                        </optgroup>
                        <optgroup label="West South Central Division">
                            <option value="Oklahoma">Oklahoma</option>
                            <option value="Texas">Texas</option>
                            <option value="Arkansas">Arkansas</option>
                            <option value="Louisiana">Louisiana</option>
                        </optgroup>
                        <optgroup label="Mountain Division">
                            <option value="Idaho">Idaho</option>
                            <option value="Montana">Montana</option>
                            <option value="Wyoming">Wyoming</option>
                            <option value="Nevada">Nevada</option>
                            <option value="Utah">Utah</option>
                            <option value="Colorado">Colorado</option>
                            <option value="Arizona">Arizona</option>
                            <option value="NewMexico">New Mexico</option>
                        </optgroup>
                        <optgroup label="Pacific Division">
                            <option value="Alaska">Alaska</option>
                            <option value="Washington">Washington</option>
                            <option value="Oregon">Oregon</option>
                            <option value="California">California</option>
                            <option value="Hawaii">Hawaii</option>
                        </optgroup>
                    </select>



                    {/* Allow users to input physical activity hours */}
                    <label className='primary_label' htmlFor="physical_activity_hours">Physical Activity Hours</label>
                    <input
                        type="number"
                        id="physical_activity_hours"
                        name="physical_activity_hours"
                        min="0"
                        value={formData.physical_activity_hours}
                        onChange={handleInputChange('physical_activity_hours')}
                        placeholder="Enter hours"
                        className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
                    />
                    <br />




                    <label className='primary_label' for="pa_cold_weather_months_per_year">Pat Cold Weather Months Per Year</label>
                    <input type="number"
                        id="pa_cold_weather_months_per_year"
                        name="pa_cold_weather_months_per_year"
                        min="0"
                        max="12"
                        placeholder="Enter pa_cold_weather_months_per_year"
                        className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md" />


                    <label className='primary_label' for="pa_hot_weather_months_per_year">Pat Hot Weather Months Per Year</label>
                    <input type="number" id="pa_hot_weather_months_per_year" name="pa_hot_weather_months_per_year" min="0" max="12" placeholder="Enter pa_hot_weather_months_per_year" className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md" />





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

                    <label className='primary_label' htmlFor="pa_surface">Surface type - where the dog plays the most</label>
                    <select
                        id="pa_surface"
                        name="pa_surface"
                        value={formData.pa_surface}
                        onChange={handleInputChange('pa_surface')}
                        className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
                    >
                        <option value="1">Concrete</option>
                        <option value="2">Wood</option>
                        <option value="3">Other hard materials</option>
                        <option value="4">Grass dirt</option>
                        <option value="5">Gravel</option>
                        <option value="6">Sand</option>
                        <option value="7">AstroTurf</option>
                    </select>



                    <label className='primary_label' for="dd_weight_lbs">Dog Weight(lbs)</label>
                    <input type="number"
                        id="dd_weight_lbs"
                        value={formData.weight}
                        onChange={handleInputChange('weight')}
                        name="dd_weight_lbs" min="0" placeholder="Enter dd_weight_lbs"
                        className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md" />



                    <label className='primary_label' for="rounded_age_years">Rounded Age years</label>
                    <input type="number"
                        id="rounded_age_years"
                        value={formData.rounded_age_years}
                        onChange={handleInputChange('rounded_age_years')}
                        name="rounded_age_years" min="0" placeholder="Enter rounded_age_years" className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md" />



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
            <div id='prediction' className="m-16 min-h-screen flex flex-column items-center justify-center">
                <div >
                    {responseData == null ? <div><h1>No Information to show.</h1></div> :
                        // <OutputInformation responseData={responseData}></OutputInformation>
                        <div>
                            <SimpleChart data={responseData}></SimpleChart>
                            <StylishBarChart data={[0.7114651843137689, 0.703769881719169, 0.7023164279123049, 0.6313146170861812, 
                            0.7402321561336324, 0.7640955968531346, 0.6557358731531935, 0.7108303529643931, 
                            0.7257548046331452, 0.7980000344784488, 0.6634657632521148, 0.6643505796408098, 
                            0.7119546231287156, 0.7045672560794011, 0.8364834828593533, 0.7696097581778122, 
                            0.6513466583661263, 0.6364062783489701, 0.5930778686116873, 0.7711782579152425]} conditionTypes={[
                            'Eye', 'Ear/Nose/Throat', 'Mouth/Dental/Oral', 'Skin', 'Cardiac', 'Respiratory',
                            'Gastrointestinal', 'Liver/Pancreas', 'Kidney/Urinary', 'Reproductive', 'Bone/Orthopedic',
                            'Brain/Neurologic', 'Endocrine', 'Hematopoietic', 'Other Congenital Disorder',
                            'Infection/Parasites', 'Toxin Consumption', 'Trauma', 'Immune-mediated', 'cancer'
                          ]} />
                        </div>
                        
                        
                    }
                </div>
            </div>
            {responseData == null ? <></> :
            <div>
                <h2 className="text-2xl mt-20 mb-20 font-semibold text-center">Health prediction for distinguished years</h2>
            </div>
            }
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4  px-6 py-3">
                {responseData == null ? (
                    <></>
                ) :  <>
                {responseData.map((item, index) => {
                    dogAge.push(item["Dog age"]);  // Fix the reference to the current item
                    return (
                        <div key={index} className="col-span-1 px-6 py-3">
                            {/* Adjust col-span and other styles as needed */}
                            <OutputInformation responseData={item} />
                        </div>
                    );
                })}
            </>
            }
            </div>

        </div>
    );
};

export default UserDataCard;



