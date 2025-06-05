import React from 'react';
import cat from '../../assets/cute-spitz.jpg'

const Overview = () => {
    return (
        <div>
            <div id='home' className="hero min-h-screen bg-base-400">
                <div className="hero-content flex-col lg:flex-row-reverse">
                    <img src={cat} className="max-w-lg rounded-lg shadow-2xl" />
                    <div>
                        <h1 className="text-6xl font-bold">Health Prediction for Dogs</h1>
                        <p className="py-6">With the development of society, dogs play an
                            increasingly important role in people‘s lives, which
                            makes people pay more attention to the health of
                            dogs. The primary problem is the challenge in effectively
                            predicting and managing health issues in dogs. This
                            includes early detection of diseases and
                            understanding breed-specific vulnerabilities.</p>
                        {/* <button className="btn btn-primary">Get Started</button> */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Overview;