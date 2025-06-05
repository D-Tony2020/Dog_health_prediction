import React from 'react';
import { Line } from 'react-chartjs-2';

const MultiLineGraph = ({ data }) => {
  const labels = data.map(entry => entry['Dog age']);

  const diseases = ['Eye', 'Ear/Nose/Throat', 'Mouth/Dental/Oral', 'Skin', 'Cardiac', 'Respiratory',
  'Gastrointestinal', 'Liver/Pancreas', 'Kidney/Urinary', 'Reproductive', 'Bone/Orthopedic',
  'Brain/Neurologic', 'Endocrine', 'Hematologic','Other Congenital Disorder',
  'Infection/Parasites', 'Toxin Consumption', 'Trauma', 'Immune-mediated', 'Cancer'];

  const datasets = diseases.map((disease, index) => ({
    label: disease,
    borderColor: `hsl(${(index / diseases.length) * 360}, 100%, 50%)`, // Assign unique color to each line
    data: data.map(entry => entry['outcome_list'][index]),
    fill: false,
  }));

  const chartData = {
    labels: labels,
    datasets: datasets,
  };

  const options = {
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'Dog Age',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Disease Probabilities',
        },
        ticks: {
          callback: function (value) {
            return (value*100).toFixed(2) + '%'; // Format tick values as percentage
          },
        },
      },
      
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: (context) => {
            const datasetLabel = context.dataset.label || '';
            const tooltipValue = (context.formattedValue*100).toFixed(2) || '';
            const dataIndex = context.dataIndex;
            const disease = diseases[dataIndex];
  
            // Customize the tooltip label as needed
            return `${datasetLabel}: ${tooltipValue}%`;
          },
        },
      },
    },

  };

  return (
    <div style={{ width: '900px', height: '700px' }}>
      <h2 className="text-2xl mt-20 mb-20 font-semibold text-center">Overall Prediction Graph </h2>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default MultiLineGraph;
