import React from 'react';
import { Bar } from 'react-chartjs-2';

const StylishBarChart = ({ data, conditionTypes }) => {
  const chartData = {
    labels: conditionTypes,
    datasets: [
      {
        label: 'Disease Accuracy score',
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)',
          'rgba(255, 206, 86, 0.2)',
          'rgba(75, 192, 192, 0.2)',
          'rgba(153, 102, 255, 0.2)',
          'rgba(255, 159, 64, 0.2)',
          // Add more custom colors as needed
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)',
          // Add more custom colors as needed
        ],
        borderWidth: 1,
        hoverBackgroundColor: [
          'rgba(255, 99, 132, 0.4)',
          'rgba(54, 162, 235, 0.4)',
          'rgba(255, 206, 86, 0.4)',
          'rgba(75, 192, 192, 0.4)',
          'rgba(153, 102, 255, 0.4)',
          'rgba(255, 159, 64, 0.4)',
          // Add more custom colors as needed
        ],
        hoverBorderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)',
          // Add more custom colors as needed
        ],
        data: data.map(value => (value * 100).toFixed(2)), // Multiply by 100 to show percentages
      },
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        max: 100, // Set max value to 100 for percentages
        title: {
          display: true,
          text: 'Probability (%)',
        },
      },
    },
  };

  return (
    <div  style={{ width: '900px', height: '700px' }}>
      <h2 className="text-2xl mt-20 mb-20 font-semibold text-center">Accuracy Score of our system</h2>
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default StylishBarChart;
