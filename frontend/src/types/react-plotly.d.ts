declare module 'react-plotly.js' {
  import * as React from 'react';
  export interface PlotProps {
    data: any[];
    layout: any;
    config?: any;
    frames?: any[];
    style?: React.CSSProperties;
    onInitialized?: (figure: any, graphDiv: any) => void;
    onUpdate?: (figure: any, graphDiv: any) => void;
    onPurge?: (component: any) => void;
    onError?: (err: any) => void;
    useResizeHandler?: boolean;
    className?: string;
    [key: string]: any; // Allow any other event handlers or parameters
  }
  export default class Plot extends React.Component<PlotProps> {}
}
