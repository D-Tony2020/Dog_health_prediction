
import './App.css'

function App() {

  return (
    <>
     <h1>Hello world</h1>
     {/* <div>
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
    </div> */}
    </>
  )
}

export default App
