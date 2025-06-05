

import { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const OutputInformation = ({ responseData }) => {
  const chartRef = useRef(null);
  const data = responseData.outcome_list;
 

  const age = responseData['Dog age']

  useEffect(() => {
    let newChartInstance = null;

    if (chartRef.current) {
      const ctx = chartRef.current.getContext('2d');

      // Destroy existing chart before creating a new one
      if (chartRef.current.chartInstance) {
        chartRef.current.chartInstance.destroy();
      }

      // Create a new chart instance
      newChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['Eye', 'Ear/Nose/Throat', 'Mouth/Dental/Oral', 'Skin', 'Cardiac', 'Respiratory',
            'Gastrointestinal', 'Liver/Pancreas', 'Kidney/Urinary', 'Reproductive', 'Bone/Orthopedic',
            'Brain/Neurologic', 'Endocrine', 'Hematologic','Other Congenital Disorder',
            'Infection/Parasites', 'Toxin Consumption', 'Trauma', 'Immune-mediated', 'Cancer'],
          datasets: [{
            label: 'Diesease Information',
            data: data,
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
          }],
        },
        options: {
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                callback: function (value) {
                  return (value*100).toFixed(2) + '%'; // Format tick values as percentage
                },
              },
            },
          },
          plugins: {
            tooltip: {
              mode: 'index',
              callbacks: {
               
                label: (context) => {
                  const datasetLabel = context.dataset.label || '';
                  const value = (context.formattedValue*100).toFixed(2);
                  const tooltipValue = value || '';
                  return ` ${datasetLabel}: ${tooltipValue}%`;
                },
                afterBody: function(context) {
                  var labelValue = context[0];
                  const tooltipLabel = labelValue.label || '';
                  let customText = "";

                  switch (tooltipLabel) {
                    case "Eye":
                      customText = "- Breed factor: Spaniels and Bulldog - higher chance \n" +
                        "  Toy dogs - lower chance\n" +
                        "- Older dogs are prone to this condition\n" 
                      break;

                    case "Ear/Nose/Throat":
                      customText = "- Breed factor: Retriever and Non-sport dogs - higher chance\n" +
                        "  French Bulldog and mixed breed dogs - lower chance\n" +
                        "- Older dogs are prone to this condition";
                      break;

                    case "Mouth/Dental/Oral":
                      customText = "- Breed factor: Mastiff - higher chance\n" +
                        "  Chiheahua - lower chance\n" +
                        "- Older dogs are prone to this condition\n" +
                        "- Control weight";
                      break;

                    case "Skin":
                      customText = "- Bulldogs  - higher chance\n" +
                        "- Highly likely to be a congenital disease\n" +
                      "- Dogs living in mountain Division of U.S. are prone to this condition";
                      break;

                    case "Cardiac":
                      customText = "- Breed factor: Cavalier King Charles Spaniel dogs - higher chance\n" +
                        "  Labrador Retriever and German Shepherd dogs  - lower chance\n" +
                        "- Older dogs are prone to this condition\n" 
                      break;


                    case "Respiratory":
                      customText = "- Breed factor: French Bull and Bull dogs - higher chance\n" +
                        "  Mastiff like group - lower chance\n" +
                        "- Older dogs are prone to this condition\n" 
                      break;

                    case "Gastrointestinal":
                      customText = "- Breed factor: Poodle and Springer Spaniel - higher chance\n" +
                        "  Mixed breed dogs      - lower chance\n" 
                      break;

                    case "Liver/Pancreas":
                      customText = "- Breed factor: German Shepherd   - higher chance\n" +
                        "  Doberman Pinscher  - lower chance\n" +
                        "- Timely and regular veterinary examinations\n" +
                        "- Provide high-quality, balanced dog food to meet the nutritional needs";
                      break;

                    case "Kidney/Urinary":
                      customText = "- Sex factor: Female  - higher chance\n" +
                        "  Male - lower chance\n" +
                        "- Breed factor: Siberian Husky  - higher chance\n" +
                        "- Timely and regular veterinary examinations\n" +
                        "- Add fewer grains to dog’s main diet\n" +
                        "- Add an appropriate amount of kidney/urinary-related supplements to dogs’ daily diet";
                      break;

                    case "Reproductive":
                      customText = "- Sex factor: Female  - higher chance\n" +
                        "- Breed factor: Retriever - higher chance\n" +
                        "  mixed breed - lower chance\n" ;
                      break;

                    case "Bone/Orthopedic":
                      customText = "- Breed factor: Sub_breed Hound  - higher chance\n" +
                        "  Poodle / Terriers - lower chance\n" +
                        "- Timely and regular veterinary examinations\n" +
                        "- Eat a balanced diet and control weight\n" +
                        "- Add an appropriate amount of bone/orthopedic-related supplements to dogs’ daily diet";
                      break;


                    case "Brain/Neurologic":
                      customText = 
                      "- Breed factor: Sight Hound  - higher chance\n" +
                      "  Sub breed Golden - lower chance\n" +
                      "- Timely and regular veterinary examinations\n" +
                        "- Add an appropriate amount of brain/neurologic related supplements to dogs’ daily diet\n" +
                        "- Try to take dogs to a place with a concrete surface during activity time.";
                      break;

                    case "Endocrine":
                      customText = "- Breed factor: Dachshund - higher chance\n" +
                        "               Mixed / Lab-mix - lower chance\n" +
                        "- Timely and regular veterinary examinations\n" +
                        "- Appropriately increase dog’s daily activity time";
                      break;

                    case "Hematopoietic":
                      customText = "- Breed factor: Mixed, Labrador Rtriener - higher chance\n" +
                        " Cattle Dog - lower chance\n" +
                        "- Add an appropriate amount of hematopoietic related supplements to dogs’ daily diet";
                      break;

                    case "Other Congenital Disorder":
                      customText = "- Breed: Subbreed Golden, Ancient and Spitz  - higher chance\n" +
                        "        German Shepherd - lower chance";
                      break;

                    case "Infection/Parasites":
                      customText = "- Breed: Sub breed Sight Hound - lower chance\n" +
                        "- Weight Control, higher weight has more chance to get\n" +
                        "- More active leads higher chance to get";
                      break;

                    case "Toxin Consumption":
                      customText = "- Breed: Chihuahua - lower chance\n" +
                        "- Give the supplement should be careful";
                      break;

                    case "Trauma":
                      customText = "- Breed: Sub-breed Golden - higher chance\n" +
                        "         Cattle Dog - lower chance\n" +
                        "- Weight Control, higher weight has more chance to get\n" +
                        "- More active leads higher chance to get";
                      break;

                    case "Immune-mediated":
                      customText = "- Breed: Shepherd, Ancient and Spitz - higher chance";
                      break;

                    // Add cases for other conditions following the same pattern

                    case "Cancer":
                      customText = "Breed: Sporting dog - Lower chance,\n" +
                        "Getting older has higher chance, diagnosis more often,\n" +
                        "More active leads lower chance";
                      break;

                    // Add more cases as needed for other conditions

                    default:
                      customText = "Default custom text if the tooltipLabel doesn't match any cases";
                  }

                  return `${customText} `;
            },
              },
              
            },
          },
        },
      });

      // Save the chart instance in the ref for future use or cleanup
      chartRef.current.chartInstance = newChartInstance;
    }

    // Cleanup when the component unmounts
    return () => {
      if (newChartInstance) {
        newChartInstance.destroy();
      }
    };
  }, [responseData]);

  return (
    <div className='overflow-visible' >
      <h1>Diesease Information in age {age} </h1>
      <canvas ref={chartRef} width="400" height="200" />
    </div>
  );
};

export default OutputInformation;
