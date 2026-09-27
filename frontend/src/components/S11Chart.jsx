import {
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(Filler, LinearScale, PointElement, LineElement, Tooltip, Legend);

export default function S11Chart({ curve, thresholdDb = -10 }) {
  if (!curve.length) {
    return (
      <div className="chart-placeholder">
        <span>Courbe en attente</span>
        <p>Lancez un sweep pour visualiser la reponse frequentielle.</p>
      </div>
    );
  }

  const frequencies = curve.map((point) => point.frequency);
  const data = {
    datasets: [
      {
        label: "S11 predit",
        data: curve.map((point) => ({ x: point.frequency, y: point.s11 })),
        borderColor: "#3ae0aa",
        backgroundColor: "rgba(58, 224, 170, 0.14)",
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.12,
        fill: true,
      },
      {
        label: `Seuil ${Number(thresholdDb).toFixed(1)} dB`,
        data: [
          { x: Math.min(...frequencies), y: thresholdDb },
          { x: Math.max(...frequencies), y: thresholdDb },
        ],
        borderColor: "rgba(255, 190, 92, 0.85)",
        borderDash: [7, 6],
        borderWidth: 1.5,
        pointRadius: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 450 },
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { labels: { color: "#a9b8c8", usePointStyle: true } },
      tooltip: {
        callbacks: {
          label: (context) => `${context.parsed.y.toFixed(4)} dB`,
          title: (items) => `${items[0].parsed.x.toFixed(4)} GHz`,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Frequence (GHz)", color: "#8394a7" },
        ticks: { color: "#8394a7" },
        grid: { color: "rgba(255, 255, 255, 0.06)" },
      },
      y: {
        title: { display: true, text: "S11 (dB)", color: "#8394a7" },
        ticks: { color: "#8394a7" },
        grid: { color: "rgba(255, 255, 255, 0.06)" },
      },
    },
  };

  return (
    <div className="chart-wrap">
      <Line data={data} options={options} />
    </div>
  );
}
