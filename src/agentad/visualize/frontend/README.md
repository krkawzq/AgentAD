# AgentAD Visualizer Frontend

The browser application is implemented in React and bundled into the adjacent
`static/` directory so the Python package has no Node.js runtime dependency.

```bash
npm install
npm test
npm run build
```

The source application is split into data navigation, track inspection,
timeline interaction, and Canvas rendering modules. `static/core.js` contains
the framework-independent geometry and viewport helpers shared by the React
renderer and Node.js tests.
