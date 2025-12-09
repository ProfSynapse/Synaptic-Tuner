import { getDatasetContent } from "@/lib/actions/datasets";
import { ClientEditor } from "./client-editor";

interface EditorPageProps {
  params: {
    filename: string;
    index: string;
  };
}

export default async function EditorPage({ params }: EditorPageProps) {
  const decodedPath = decodeURIComponent(params.filename);
  const index = parseInt(params.index, 10);

  let entry;
  try {
    const data = await getDatasetContent(decodedPath);
    if (index < 0 || index >= data.length) {
      return <div>Entry not found</div>;
    }
    entry = data[index];
  } catch (e) {
    return <div>Error loading entry: {String(e)}</div>;
  }

  return (
    <ClientEditor 
      entry={entry} 
      filename={decodedPath} 
      index={index} 
    />
  );
}
