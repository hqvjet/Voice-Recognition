import axios from "axios"; 
import React, { Component } from "react";
 
class App extends Component {
    state = {
        selectedFile: null,
        serverResponse: null // Thêm state để lưu trữ dữ liệu từ server
    };
 
    onFileChange = (event) => {
        this.setState({
            selectedFile: event.target.files[0],
        });
    };
 
    onFileUpload = () => {
        const formData = new FormData();
        formData.append(
            "file",
            this.state.selectedFile,
            this.state.selectedFile.name
        );
 
        axios.post("http://192.168.33.62:8000/predict", formData)
        .then((response) => {
            this.setState({ serverResponse: response.data });
        })
        .catch((error) => {
            console.error('Error uploading file:', error);
        });
    };
 
    fileData = () => {
        if (this.state.selectedFile) {
            return (
                <div>
                    <h2>File Details:</h2>
                    <p>
                        File Name:{" "}
                        {this.state.selectedFile.name}
                    </p>
 
                    <p>
                        File Type:{" "}
                        {this.state.selectedFile.type}
                    </p>
 
                    <p>
                        Last Modified:{" "}
                        {this.state.selectedFile.lastModifiedDate.toDateString()}
                    </p>
                </div>
            );
        } else {
            return (
                <div>
                    <br />
                    <h4>
                        Choose before Pressing the Upload
                        button
                    </h4>
                </div>
            );
        }
    };
 
    render() {
        return (
            <div>
                <h1>Voice Recognition</h1>
                <h3>File Upload using React!</h3>
                <div>
                    <input
                        type="file"
                        onChange={this.onFileChange}
                    />
                    <button onClick={this.onFileUpload}>
                        Upload!
                    </button>
                </div>
                {this.fileData()}
                {this.state.serverResponse && (
                    <div>
                        <h2>Server Response:</h2>
                        <p>{this.state.serverResponse}</p>
                    </div>
                )}
            </div>
        );
    }
}
 
export default App;
