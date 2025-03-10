import React from 'react'
import Plot from 'react-plotly.js'

export default class App extends React.Component {

  constructor(){
    super();

    this.state = {response: null}

    this.buildPlots = this.buildPlots.bind(this)
  }

  componentDidMount(){
    const socket = new WebSocket("ws://localhost:8080")
    socket.onmessage = (evt) => {
      const resp = JSON.parse(evt.data)
      this.setState({ response: resp })
    }
  }

  buildPlots() {
    const hold = []
    const { response } = this.state
    if(response !== null){
      Object.keys(response).map((key, id) => {
        hold.push(
          <Plot
            data={[{
              x: response[key]['ClassA']['x'],
              y: response[key]['ClassA']['y'],
              z: response[key]['ClassA']['z'],
              type: 'scatter3d',
              mode: 'markers',
              marker: {
                color: response[key]['ClassA']['color'],
                size: 4
              }
            },{
              x: response[key]['ClassB']['x'],
              y: response[key]['ClassB']['y'],
              z: response[key]['ClassB']['z'],
              type: 'scatter3d',
              mode: 'markers',
              marker: {
                color: response[key]['ClassB']['color'],
                size: 4
              }
            },{
              x: response[key]['Surface']['x'],
              y: response[key]['Surface']['y'],
              z: response[key]['Surface']['z'],
              type: 'surface',
              colorscale: 'Greys', // Single-color scale
              showscale: false
            }]}
            layout={{
              title: key + ' | ' + response[key]['Title']
            }}
          />
        )
      })
    }

    return hold 
  }

  render(){

    return (
      <div>{this.buildPlots()}</div>
    );
  }

}