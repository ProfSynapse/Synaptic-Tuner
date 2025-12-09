import { getDatasetContent } from "@/lib/actions/datasets";
import { DatasetTableWrapper } from "./client-wrapper";

interface DatasetPageProps {
  params: {
    filename: string;
  };
}

export default async function DatasetPage({ params }: DatasetPageProps) {
  const decodedPath = decodeURIComponent(params.filename);
  
  let data;
  try {
    data = await getDatasetContent(decodedPath);
  } catch (e) {
    return <div>Error loading dataset: {String(e)}</div>;
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dataset Explorer</h1>
        <p className="text-muted-foreground break-all">
          {decodedPath}
        </p>
      </div>
      
      <DatasetTableWrapper data={data} filename={decodedPath} />
    </div>
  );
}
