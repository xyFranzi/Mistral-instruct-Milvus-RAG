from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pymilvus import MilvusClient
import os

db_path = os.path.expanduser("~/mypython/Mistral-instruct-Milvus-RAG/milvus_german_docs.db")
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Load German_Semantic_V3b embedding model
hf_embeddings = HuggingFaceEmbeddings(
    model_name="aari1995/German_Semantic_V3b",
    model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
)

# Sample documents
docs = [
    Document(page_content="Der Roman erzählt die Geschichte eines introvertierten Mannes namens Meursault. Er hat einen Totschlag begangen und wartet in seiner Gefängniszelle auf die Hinrichtung. Die Handlung spielt im Algerien der 1930er Jahre. Der Name Meursault ist möglicherweise von „Meurs, sot!“, zu Deutsch etwa „Stirb, (du) Trottel!“, abgeleitet.", metadata={"source": "der_fremde_wiki_inhalt_0.pdf"}),
    Document(page_content="Der Roman ist in zwei Teile gegliedert. Der erste Teil beginnt mit den Worten: „Heute ist Mama gestorben. Oder vielleicht gestern, ich weiß es nicht.“ Bei der Beerdigung seiner Mutter zeigt Meursault keine Emotionen. Die fehlende Anteilnahme beruht offenbar auf einem kühlen Verhältnis, das zwischen Mutter und Sohn herrschte. Der Roman setzt sich mit einer Dokumentation der folgenden Tage von Meursaults Leben aus der Ich-Perspektive fort.", metadata={"source": "der_fremde_wiki_inhalt_1.pdf"}),
    Document(page_content="Meursault zeigt sich als Mensch, der antriebslos in den Tag hineinlebt, der zwar Details seiner Umgebung wahrnimmt, jedoch Gewalt und Ungerechtigkeit ungerührt hinnimmt. Kurz nach der Beerdigung seiner Mutter beginnt er eine Liebesaffäre, was später als Beweis für seine emotionale Kälte angeführt wird. Meursault ist offenbar zufrieden, wenn sein Alltag routinemäßig und wie gewohnt verläuft.", metadata={"source": "der_fremde_wiki_inhalt_2.pdf"}),
    Document(page_content="Sein Nachbar, Raymond Sintès, der der Zuhälterei verdächtigt wird, freundet sich mit ihm an. Meursault hilft Raymond, eine Mätresse, eine Araberin, die er als ehemalige Freundin ausgibt, anzulocken. Raymond bedrängt und demütigt die Frau. Später begegnen Meursault und Raymond dem Bruder der Frau und dessen Freunden am Strand, es kommt zu einer Schlägerei. Kurz danach trifft Meursault einen der Araber wieder, der bei seinem Anblick ein Messer zieht. Vom Glanz der Sonne auf der Messerklinge geblendet, umklammert Meursault in seiner Jackentasche einen von Raymond ausgeliehenen Revolver, zückt die Waffe und tötet den Araber mit einem Schuss. Ohne besonderen Grund gibt er unmittelbar darauf vier weitere Schüsse auf den Leichnam ab, was vor Gericht zum Ausschluss von Notwehr und unbeabsichtigtem Totschlag und letztlich zur Verurteilung Meursaults als Mörder führt. Meursaults mögliche Unzurechnungsfähigkeit nach Stunden in praller Sonne steht im Raum.", metadata={"source": "der_fremde_wiki_inhalt_3.pdf"}),
    Document(page_content="Heute ist Mama gestorben. Vielleicht auch gestern, ich weiß nicht. Ich habe ein Telegramm vom Heim bekommen: «Mutter verstorben. Beisetzung morgen. Hochachtungsvoll.» Das will nichts heißen. Es war vielleicht gestern. Das Altersheim ist in Marengo, achtzig Kilometer von Algier entfernt. Ich werde den Bus um zwei nehmen und nachmittags ankommen. Auf die Weise kann ich Totenwache halten und bin morgen Abend wieder zurück. Ich habe meinen Chef um zwei Tage Urlaub gebeten, und bei so einem Entschuldigungsgrund konnte er sie mir nicht abschlagen. Aber er sah nicht erfreut aus. Ich habe sogar gesagt: «Es ist nicht meine Schuld.» Er hat nicht geantwortet. Da habe ich gedacht, dass ich das nicht hätte sagen sollen. Ich brauchte mich ja nicht zu entschuldigen. Vielmehr hätte er mir sein Beileid aussprechen müssen. Aber das wird er wahrscheinlich übermorgen tun, wenn er mich in Trauer sieht. Vorläufig ist es ein bisschen so, als wäre Mama gar nicht tot. Nach der Beerdigung allerdings wird es eine abgeschlossene Sache sein, und alles wird einen offizielleren Anstrich bekommen haben.", metadata={"source": "der_fremde_ch1_1.pdf"}),
    Document(page_content="Ich habe den Bus um zwei genommen. Es war sehr heiß. Ich habe im Restaurant von Céleste gegessen, wie gewöhnlich. Sie hatten alle viel Mitgefühl mit mir, und Céleste hat gesagt: «Man hat nur eine Mutter.» Als ich gegangen bin, haben sie mich zur Tür begleitet. Ich war etwas abgelenkt, weil ich noch zu Emmanuel hinaufmusste, um mir einen schwarzen Schlips und eine Trauerbinde von ihm zu borgen. Er hat vor ein paar Monaten seinen Onkel verloren. Ich bin gelaufen, um den Bus nicht zu verpassen. Diese Hetze, dieses Laufen – wahrscheinlich war es all das, zusammen mit dem Gerüttel, dem Benzingeruch, der Spiegelung der Straße und des Himmels, weswegen ich eingenickt bin. Ich habe fast während der ganzen Fahrt geschlafen. Und als ich aufgewacht bin, war ich gegen einen Soldaten gerutscht, der mich angelächelt hat und gefragt hat, ob ich von weit her käme. Ich habe «ja» gesagt, um nicht weiterreden zu müssen.", metadata={"source": "der_fremde_ch1_2.pdf"}),
]

client = MilvusClient(db_path)
collection_name = "german_docs"

# create collection if it doesn't exist
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# Get embedding dimension
sample_embedding = hf_embeddings.embed_query("test")
dim = len(sample_embedding)

client.create_collection(
    collection_name=collection_name,
    dimension=dim,
    metric_type="IP",  # Inner Product (cosine similarity)
    consistency_level="Strong"
)

data = []
for i, doc in enumerate(docs):
    embedding = hf_embeddings.embed_query(doc.page_content)
    data.append({
        "id": i,
        "text": doc.page_content,
        "source": doc.metadata.get("source", "unknown"),
        "vector": embedding
    })

# Insert documents
client.insert(collection_name=collection_name, data=data)

print(f"✓ Embedding done - data stored in: {db_path}")
print(f"✓ Added {len(docs)} documents to collection '{collection_name}'")

if os.path.exists(db_path):
    print(f"✓ Database file size: {os.path.getsize(db_path)} bytes")

# Example search
def search_documents(query, top_k=3):
    query_embedding = hf_embeddings.embed_query(query)
    
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source"]
    )
    
    return results[0]  # Return first (and only) query results

# Example search
query = "Wer ist Raymond?"
results = search_documents(query, top_k=3)

print(f"\nSuche nach: '{query}'")
print("-" * 50)

for i, result in enumerate(results, 1):
    print(f"Ergebnis {i}:")
    print(f"Text: {result['entity']['text']}")
    print(f"Quelle: {result['entity']['source']}")
    print(f"Ähnlichkeit: {result['distance']:.4f}")
    print("-" * 30)

# Close client
client.close()