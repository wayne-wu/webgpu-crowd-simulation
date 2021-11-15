export const getAgentData = (numAgents: number) => {
  const agentIdxOffset = 12;
  const initialAgentData = new Float32Array(numAgents * agentIdxOffset);  //48 is total byte size of each agent

  for (let i = 0; i < numAgents/2; ++i) {
    // position.xyz
    initialAgentData[agentIdxOffset * i + 0] = 10 * (Math.random() - 0.5);
    initialAgentData[agentIdxOffset * i + 1] = 0.1;
    initialAgentData[agentIdxOffset * i + 2] = 10 + 2 * (Math.random() - 0.5);

    // color.rgba
    initialAgentData[agentIdxOffset * i + 4] = 1;
    initialAgentData[agentIdxOffset * i + 5] = 0;
    initialAgentData[agentIdxOffset * i + 6] = 0;
    initialAgentData[agentIdxOffset * i + 7] = 1;

    // velocity.xyz
    initialAgentData[agentIdxOffset * i + 8] = 0;
    initialAgentData[agentIdxOffset * i + 9] = 0;
    initialAgentData[agentIdxOffset * i + 10] = (0.1 + 0.5 * Math.random())*-1;
  }
  for (let i = numAgents/2; i < numAgents; ++i) {
    // position.xyz
    initialAgentData[agentIdxOffset * i + 0] = 10 * (Math.random() - 0.5);
    initialAgentData[agentIdxOffset * i + 1] = 0.1;
    initialAgentData[agentIdxOffset * i + 2] = -10 + 2 * (Math.random() - 0.5);

    // color.rgba
    initialAgentData[agentIdxOffset * i + 4] = 0;
    initialAgentData[agentIdxOffset * i + 5] = 0;
    initialAgentData[agentIdxOffset * i + 6] = 1;
    initialAgentData[agentIdxOffset * i + 7] = 1;

    // velocity.xyz
    initialAgentData[agentIdxOffset * i + 8] = 0;
    initialAgentData[agentIdxOffset * i + 9] = 0;
    initialAgentData[agentIdxOffset * i + 10] = (0.1 + 0.5 * Math.random());
  }
  return initialAgentData;
};