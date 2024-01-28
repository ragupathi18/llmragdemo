import chromadb
import csv
def getContext():
	chroma_client= chromadb.PersistentClient(path="./chromadb")

#collection=chroma_client.delete_collection(name="countries")
	try:
		collection=chroma_client.create_collection(name="countries")
		print("Created Countries collection ")
	    #chroma_client.delete_collection("countries")
	except chromadb.db.base.UniqueConstraintError:
		print("Countries collection already exists, deleting")
		chroma_client.delete_collection(name="countries")
		collection=chroma_client.create_collection(name="countries")
		print("Created Countries collection ")

	
	with open("context_help.txt",mode='r') as f:
		lines=list(csv.reader(f))
	i=0
	for line in lines:
		i+=1

		collection.upsert(documents=line[1],metadatas=[{"country":line[0]}],ids=[str(i)])    
#	print(collection.peek())
#getContext()