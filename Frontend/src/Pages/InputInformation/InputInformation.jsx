
import React, { useState } from 'react';
import { useEffect } from 'react';
import OutputInformation from '../OutputInformation/OutputInformation';
import SimpleChart from '../OutputInformation/SimpleChart';

const UserDataCard = () => {
  const [formData, setFormData] = useState({
    primaryBreed: '',
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

    // Add values for all primary breeds
    const primaryBreeds = document.querySelectorAll("#primaryBreed option");
    primaryBreeds.forEach((option) => {
      featureList.push((option.value === formData.primaryBreed) ? 1 : 0);
    });

    // Add values for breedGroups
    const breedGroups = document.querySelectorAll("#breedGroup option");
    breedGroups.forEach((option) => {
      featureList.push((option.value === formData.breedGroup ? true : false));
    });

    // Add values for all sub-breeds
    const subBreeds = document.querySelectorAll("#subBreed option");
    subBreeds.forEach((option) => {
      featureList.push((option.value === formData.subBreed ? true : false));
    });

    // Add values for breedType
    const breedType = document.querySelectorAll("#breedType option");
    breedType.forEach((option) => {
      featureList.push((option.value === formData.breedType ? true : false));
    });

    // Add values for breed_pure_or_mix
    const breedPureOrMixValue = formData.breed_pure_or_mix;
    featureList.push((breedPureOrMixValue === "1") ? 1 : 0);

    // Add values for sex
    const sex = document.querySelectorAll("#sex option");
    sex.forEach((option) => {
      featureList.push((option.value === formData.sex ? true : false));
    });

    // Add values for ColdWheater_with_id
    const ColdWheater_with_id = document.querySelectorAll("#ColdWheater_with_id option");
    ColdWheater_with_id.forEach((option) => {
      featureList.push((option.value === formData.ColdWheater_with_id) ? 1 : 0);
    });

    // Add value for physical_activity_hours
    const physicalActivityHoursValue = formData.physical_activity_hours_input;
    featureList.push(parseFloat(physicalActivityHoursValue));

    // Add values for pa_surface
    const pa_surface = document.querySelectorAll("#pa_surface option");
    pa_surface.forEach((option) => {
      featureList.push((option.value === formData.pa_surface ? true : false));
    });

    // Log the featureList (for testing purposes)
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
      body: JSON.stringify({ featureList: featureList }),
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

          <label htmlFor="primaryBreed">Choose primary Breed：</label>
          <select
            id="primaryBreed"
            name="primaryBreed"
            value={formData.primaryBreed}
            onChange={handleInputChange('primaryBreed')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1">American Pitbull Terrier</option>
            <option value="2">American Staffordshire Terrier</option>
            <option value="3">Australian Cattle Dog</option>
            <option value="4">Australian Shepherd</option>
            <option value="5">Basset Hound</option>
            <option value="6">Beagle</option>
            <option value="7">Bernese Mountain Dog</option>
            <option value="8">Bichon Frise</option>
            <option value="9">Border Collie</option>
            <option value="10">Boston Terrier</option>
            <option value="11">Boxer</option>
            <option value="12">Bulldog</option>
            <option value="13">Cairn Terrier</option>
            <option value="14">Catahoula Leopard Dog</option>
            <option value="15">Cavalier King Charles Spaniel</option>
            <option value="16">Chihuahua</option>
            <option value="17">Chow Chow</option>
            <option value="18">Cocker Spaniel</option>
            <option value="19">Collie</option>
            <option value="20">Dachshund</option>
            <option value="21">Doberman Pinscher</option>
            <option value="22">English Springer Spaniel</option>
            <option value="23">French Bulldog</option>
            <option value="24">German Shepherd Dog</option>
            <option value="25">German Shorthaired Pointer</option>
            <option value="26">Golden Retriever</option>
            <option value="27">Great Dane</option>
            <option value="28">Great Pyrenees</option>
            <option value="29">Greyhound</option>
            <option value="30">Havanese</option>
            <option value="31">Jack Russell Terrier</option>
            <option value="32">Labrador Retriever</option>
            <option value="33">Maltese</option>
            <option value="34">Miniature Pinscher</option>
            <option value="35">Miniature Schnauzer</option>
            <option value="36">Newfoundland</option>
            <option value="37">Other</option>
            <option value="38">Pembroke Welsh Corgi</option>
            <option value="39">Pomeranian</option>
            <option value="40">Poodle</option>
            <option value="41">Poodle (Toy)</option>
            <option value="42">Pug</option>
            <option value="43">Rat Terrier</option>
            <option value="44">Rhodesian Ridgeback</option>
            <option value="45">Rottweiler</option>
            <option value="46">Shetland Sheepdog</option>
            <option value="47">Shih Tzu</option>
            <option value="48">Siberian Husky</option>
            <option value="49">West Highland White Terrier</option>
            <option value="50">Yorkshire Terrier</option>
          </select>

          <br />

          <label htmlFor="breedGroup">Breed Group:</label>
          <select
            id="breedGroup"
            name="breedGroup"
            value={formData.breedGroup}
            onChange={handleInputChange('breedGroup')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >

            <option value="1">Ancient and Spitz</option>
            <option value="2">Australian-like</option>
            <option value="3">Chihuahua</option>
            <option value="4">Dachshund</option>
            <option value="5">Golden</option>
            <option value="6">Herding dogs - Other</option>
            <option value="7">Hound</option>
            <option value="8">Labs</option>
            <option value="9">Mastiff-like Group 1</option>
            <option value="10">Mastiff-like Group 2</option>
            <option value="11">Mixed Lab and Golden</option>
            <option value="12">Mixed Large</option>
            <option value="13">Mixed Medium</option>
            <option value="14">Mixed Other</option>
            <option value="15">Mixed Small</option>
            <option value="16">Shepherd</option>
            <option value="17">Spaniels</option>
            <option value="18">Terriers</option>
            <option value="19">Toy - Other</option>
            <option value="20">Working dogs - Non-sport</option>
          </select>

          <br />

          <label htmlFor="subBreed">Choose sub breed：</label>
          <select
            id="subBreed"
            name="subBreed"
            value={formData.subBreed}
            onChange={handleInputChange('subBreed')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1">True Mastiffs</option>
            <option value="2">Ancient and Spitz</option>
            <option value="3">Australian</option>
            <option value="4">Bulldogs</option>
            <option value="5">Chihuahua</option>
            <option value="6">Collie, Corgi, Sheepdog</option>
            <option value="7">Dachshund</option>
            <option value="8">Golden</option>
            <option value="9">Golden mix</option>
            <option value="10">Herding-other</option>
            <option value="11">Hound</option>
            <option value="12">Lab mix</option>
            <option value="13">Labs</option>
            <option value="14">Large</option>
            <option value="15">Large Terriers</option>
            <option value="16">Maltese</option>
            <option value="17">Medium</option>
            <option value="18">Non-sport</option>
            <option value="19">Other</option>
            <option value="20">Pointer</option>
            <option value="21">Setter</option>
            <option value="22">Shepherd</option>
            <option value="23">Shepherd - Other</option>
            <option value="24">Shih Tzu</option>
            <option value="25">Sight Hounds</option>
            <option value="26">Similar to retrievers</option>
            <option value="27">Small</option>
            <option value="28">Small Terriers</option>
            <option value="29">Sporting</option>
            <option value="30">Toy - Other</option>
          </select>

          <br />

          <label htmlFor="breedType">Breed Type:</label>
          <select
            id="breedType"
            name="breedType"
            value={formData.breedType}
            onChange={handleInputChange('breedType')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1">Ancient and Spitz</option>
            <option value="2">Herding dogs</option>
            <option value="3">Mastiff-Like</option>
            <option value="4">Mixed</option>
            <option value="5">Retriever</option>
            <option value="6">Scent Hounds</option>
            <option value="7">Spaniels</option>
            <option value="8">Terriers</option>
            <option value="9">Toy dogs</option>
            <option value="10">Working dogs</option>
          </select>

          <br />

          <label htmlFor="breed_pure_or_mix">Breed Pure or Mix:</label>
          <select
            id="breed_pure_or_mix"
            name="breed_pure_or_mix"
            value={formData.breed_pure_or_mix}
            onChange={handleInputChange('breed_pure_or_mix')}
            className="w-full px-3 py-2 mb-3 border border-gray-300 rounded-md"
          >
            <option value="1">Pure</option>
            <option value="2">Mixed</option>
          </select>

          <br />

          <label htmlFor="sex">Sex:</label>
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

          <br />

          <label htmlFor="ColdWheater_with_id">Cold Weather with ID:</label>
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

          <br />

          {/* Allow users to input physical activity hours */}
          <label htmlFor="physical_activity_hours_input">Physical Activity Hours:</label>
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

          <label htmlFor="pa_surface">Floor type:</label>
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



// import React, { useState } from 'react';

// const UserDataCard = () => {
//   const [formData, setFormData] = useState({
//     breed_pure_or_mix: 'pure',
//     weight: 'small',
//     primaryBreed: '',
//     secondaryBreed: '',
//     sex: '1',
//     ColdWheater_with_id: '1.0',
//     physical_activity_hours: '',
//     pa_surface: '1',
//   });

//   const handleInputChange = (event) => {
//     const { name, value } = event.target;
//     setFormData({ ...formData, [name]: value });
//   };

//   const generateFeatureListAndSubmit = () => {
//     // Create an array to hold the feature values
//     const featureList = [];

//     // Add values for all primary breeds
//     const primaryBreeds = document.querySelectorAll("#primaryBreed option");
//     primaryBreeds.forEach((option) => {
//       featureList.push((option.value === formData.primaryBreed) ? 1 : 0);
//     });

//     // Add values for breed_pure_or_mix
//     const breedPureOrMixValue = formData.breed_pure_or_mix;
//     featureList.push((breedPureOrMixValue === "pure") ? 1 : 0);

//     // Add values for sex
//     const sex = document.querySelectorAll("#sex option");
//     sex.forEach((option) => {
//       featureList.push((option.value === formData.sex ? true : false));
//     });

//     // Add values for ColdWheater_with_id
//     const ColdWheater_with_id = document.querySelectorAll("#ColdWheater_with_id option");
//     ColdWheater_with_id.forEach((option) => {
//       featureList.push((option.value === formData.ColdWheater_with_id) ? 1 : 0);
//     });

//     // Add value for physical_activity_hours
//     const physicalActivityHoursValue = formData.physical_activity_hours;
//     featureList.push(parseFloat(physicalActivityHoursValue));

//     // Add values for pa_surface
//     const pa_surface = document.querySelectorAll("#pa_surface option");
//     pa_surface.forEach((option) => {
//       featureList.push((option.value === formData.pa_surface ? true : false));
//     });

//     // Log the featureList (for testing purposes)
//     console.log(featureList);

//     // Set the featureList in the hidden input field
//     const featureListJSON = JSON.stringify(featureList);
//     console.log(featureListJSON);

//     // Make an HTTP POST request or handle the featureList as needed
//     // ...

//     // Additional logic as needed
//     // ...
//   };

//   return (
//     <div align="center">
//       <form>
//         {/* ... (Other form elements) ... */}

//         <label htmlFor="breed_pure_or_mix">Breed Pure or Mix:</label>
//         <select
//           id="breed_pure_or_mix"
//           name="breed_pure_or_mix"
//           value={formData.breed_pure_or_mix}
//           onChange={handleInputChange}
//         >
//           <option value="pure">pure</option>
//           <option value="mixed">mixed</option>
//         </select>

//         <label for="primaryBreed">Choose primary Breed：</label>
//             <select id="primaryBreed" 
//             name="primaryBreed"
//             value={formData.primaryBreed}
//             onChange={handleInputChange}
//             >
            
//                 <option value="American Pitbull Terrier">American Pitbull Terrier</option>
//                 <option value="American Staffordshire Terrier">American Staffordshire Terrier</option>
//                 <option value="Australian Cattle Dog">Australian Cattle Dog</option>
//                 <option value="Australian Shepherd">Australian Shepherd</option>
//                 <option value="Basset Hound">Basset Hound</option>
//                 <option value="Beagle">Beagle</option>
//                 <option value="Bernese Mountain Dog">Bernese Mountain Dog</option>
//                 <option value="Bichon Frise">Bichon Frise</option>
//                 <option value="Border Collie">Border Collie</option>
//                 <option value="Boston Terrier">Boston Terrier</option>
//                 <option value="Boxer">Boxer</option>
//                 <option value="Bulldog">Bulldog</option>
//                 <option value="Cairn Terrier">Cairn Terrier</option>
//                 <option value="Catahoula Leopard Dog">Catahoula Leopard Dog</option>
//                 <option value="Cavalier King Charles Spaniel">Cavalier King Charles Spaniel</option>
//                 <option value="Chihuahua">Chihuahua</option>
//                 <option value="Chow Chow">Chow Chow</option>
//                 <option value="Cocker Spaniel">Cocker Spaniel</option>
//                 <option value="Collie">Collie</option>
//                 <option value="Dachshund">Dachshund</option>
//                 <option value="Doberman Pinscher">Doberman Pinscher</option>
//                 <option value="English Springer Spaniel">English Springer Spaniel</option>
//                 <option value="French Bulldog">French Bulldog</option>
//                 <option value="German Shepherd Dog">German Shepherd Dog</option>
//                 <option value="German Shorthaired Pointer">German Shorthaired Pointer</option>
//                 <option value="Golden Retriever">Golden Retriever</option>
//                 <option value="Great Dane">Great Dane</option>
//                 <option value="Great Pyrenees">Great Pyrenees</option>
//                 <option value="Greyhound">Greyhound</option>
//                 <option value="Havanese">Havanese</option>
//                 <option value="Jack Russell Terrier">Jack Russell Terrier</option>
//                 <option value="Labrador Retriever">Labrador Retriever</option>
//                 <option value="Maltese">Maltese</option>
//                 <option value="Miniature Pinscher">Miniature Pinscher</option>
//                 <option value="Miniature Schnauzer">Miniature Schnauzer</option>
//                 <option value="Newfoundland">Newfoundland</option>
//                 <option value="Other">Other</option>
//                 <option value="Pembroke Welsh Corgi">Pembroke Welsh Corgi</option>
//                 <option value="Pomeranian">Pomeranian</option>
//                 <option value="Poodle">Poodle</option>
//                 <option value="Poodle (Toy)">Poodle (Toy)</option>
//                 <option value="Pug">Pug</option>
//                 <option value="Rat Terrier">Rat Terrier</option>
//                 <option value="Rhodesian Ridgeback">Rhodesian Ridgeback</option>
//                 <option value="Rottweiler">Rottweiler</option>
//                 <option value="Shetland Sheepdog">Shetland Sheepdog</option>
//                 <option value="Shih Tzu">Shih Tzu</option>
//                 <option value="Siberian Husky">Siberian Husky</option>
//                 <option value="West Highland White Terrier">West Highland White Terrier</option>
//                 <option value="Yorkshire Terrier">Yorkshire Terrier</option>
//                 <option value="Unknown">Unknown</option>
//             </select>

        

//             <label for="secondaryBreed">Choose secondary Breed：</label>
//             <select id="secondaryBreed"
//             name="secondaryBreed"
//             value={formData.secondaryBreed}
//             onChange={handleInputChange}>
//                 <option value="American Pitbull Terrier">American Pitbull Terrier</option>
//                 <option value="American Staffordshire Terrier">American Staffordshire Terrier</option>
//                 <option value="Australian Cattle Dog">Australian Cattle Dog</option>
//                 <option value="Australian Shepherd">Australian Shepherd</option>
//                 <option value="Basset Hound">Basset Hound</option>
//                 <option value="Beagle">Beagle</option>
//                 <option value="Bernese Mountain Dog">Bernese Mountain Dog</option>
//                 <option value="Bichon Frise">Bichon Frise</option>
//                 <option value="Border Collie">Border Collie</option>
//                 <option value="Boston Terrier">Boston Terrier</option>
//                 <option value="Boxer">Boxer</option>
//                 <option value="Bulldog">Bulldog</option>
//                 <option value="Cairn Terrier">Cairn Terrier</option>
//                 <option value="Catahoula Leopard Dog">Catahoula Leopard Dog</option>
//                 <option value="Cavalier King Charles Spaniel">Cavalier King Charles Spaniel</option>
//                 <option value="Chihuahua">Chihuahua</option>
//                 <option value="Chow Chow">Chow Chow</option>
//                 <option value="Cocker Spaniel">Cocker Spaniel</option>
//                 <option value="Collie">Collie</option>
//                 <option value="Dachshund">Dachshund</option>
//                 <option value="Doberman Pinscher">Doberman Pinscher</option>
//                 <option value="English Springer Spaniel">English Springer Spaniel</option>
//                 <option value="French Bulldog">French Bulldog</option>
//                 <option value="German Shepherd Dog">German Shepherd Dog</option>
//                 <option value="German Shorthaired Pointer">German Shorthaired Pointer</option>
//                 <option value="Golden Retriever">Golden Retriever</option>
//                 <option value="Great Dane">Great Dane</option>
//                 <option value="Great Pyrenees">Great Pyrenees</option>
//                 <option value="Greyhound">Greyhound</option>
//                 <option value="Havanese">Havanese</option>
//                 <option value="Jack Russell Terrier">Jack Russell Terrier</option>
//                 <option value="Labrador Retriever">Labrador Retriever</option>
//                 <option value="Maltese">Maltese</option>
//                 <option value="Miniature Pinscher">Miniature Pinscher</option>
//                 <option value="Miniature Schnauzer">Miniature Schnauzer</option>
//                 <option value="Newfoundland">Newfoundland</option>
//                 <option value="Other">Other</option>
//                 <option value="Pembroke Welsh Corgi">Pembroke Welsh Corgi</option>
//                 <option value="Pomeranian">Pomeranian</option>
//                 <option value="Poodle">Poodle</option>
//                 <option value="Poodle (Toy)">Poodle (Toy)</option>
//                 <option value="Pug">Pug</option>
//                 <option value="Rat Terrier">Rat Terrier</option>
//                 <option value="Rhodesian Ridgeback">Rhodesian Ridgeback</option>
//                 <option value="Rottweiler">Rottweiler</option>
//                 <option value="Shetland Sheepdog">Shetland Sheepdog</option>
//                 <option value="Shih Tzu">Shih Tzu</option>
//                 <option value="Siberian Husky">Siberian Husky</option>
//                 <option value="West Highland White Terrier">West Highland White Terrier</option>
//                 <option value="Yorkshire Terrier">Yorkshire Terrier</option>
//                 <option value="Unknown">Unknown</option>
//             </select>

//         <label htmlFor="sex">Sex:</label>
//         <select
//           id="sex"
//           name="sex"
//           value={formData.sex}
//           onChange={handleInputChange}
//         >
//           <option value="1">Female</option>
//           <option value="2">Male</option>
//         </select>

//         {/* ... (Other form elements) ... */}

//         <label htmlFor="ColdWheater_with_id">Cold Weather with ID:</label>
//         <select
//           id="ColdWheater_with_id"
//           name="ColdWheater_with_id"
//           value={formData.ColdWheater_with_id}
//           onChange={handleInputChange}
//         >
//           <option value="1.0">cold_weather_1.0</option>
//           <option value="2.0">cold_weather_2.0</option>
//           <option value="3.0">cold_weather_3.0</option>
//           <option value="4.0">cold_weather_4.0</option>
//           <option value="5.0">cold_weather_5.0</option>
//         </select>

//         {/* ... (Other form elements) ... */}

//         <label htmlFor="physical_activity_hours">Physical Activity Hours:</label>
//         <input
//           type="number"
//           id="physical_activity_hours"
//           name="physical_activity_hours"
//           min="0"
//           value={formData.physical_activity_hours}
//           onChange={handleInputChange}
//           placeholder="Enter hours"
//         />

//         {/* ... (Other form elements) ... */}

//         <label htmlFor="pa_surface">Floor type:</label>
//         <select
//           id="pa_surface"
//           name="pa_surface"
//           value={formData.pa_surface}
//           onChange={handleInputChange}
//         >
//           <option value="1">concrete</option>
//           <option value="2">soft_ground</option>
//           <option value="3">unpaved_roads</option>
//           <option value="5">gravel</option>
//                 <option value="6">sand</option>
//                 <option value="7">astroturf</option>
//         </select>

//         {/* ... (Other form elements) ... */}

//         <button type="button" onClick={generateFeatureListAndSubmit}>
//           Submit
//         </button>
//       </form>
//     </div>
//   );
// };

// export default UserDataCard;
