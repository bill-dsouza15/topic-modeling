from collections import defaultdict
import pandas as pd
import numpy as np
import plotly.express as px

class TopicModel:
  def __init__(self, embedding_model, dimreduce_model, cluster_model, keyword_model, n_labels=1):
    '''
    embedding_model : Any
      A model for transforming the text into embeddings
    dimreduce_model : Any
      A model for reducing the embeddings into 2D embeddings
    cluster_model : Any
      A model for clustering the embeddings intp topics
    keyword_model : dict
      A dictionary containing a key-value pair of the model namethe model itself.
      Format: 
      {
        'KeyBert' : KeyBert model
      }
      {
        'Llama' : [pipeline, prompt]
      }
      Currently supports only KeyBert and Llama
    n_labels : int | default = 1
      Number of labels to be calculated per topic. Required for KeyBert.
    '''
    self.embedding_model = embedding_model
    self.dimreduce_model = dimreduce_model
    self.cluster_model = cluster_model
    self.keyword_model = keyword_model
    self.n_labels = n_labels
    self.key_model_name = None



  def get_embeddings(self):
    '''
    Returns original embeddings returned by embedding model
    '''
    return self.embeddings



  def get_reduced_embeddings(self):
    '''
    Returns dimensionally reduced embeddings returned by dimreduce model
    '''
    return self.reduced_embeddings
  


  def get_topics(self):
    '''
    Returns cluster topics as returned by the clustering algorithm
    '''
    return self.cluster_topics



  def get_labels(self):
    '''
    Returns cluster labels per document
    '''
    self.data_labels = None
    if self.key_model_name == "KeyBert":
      self.data_labels = [self.keyword_dict[self.cluster_topics[i]][0][0] for i in range(0, len(self.data))]
    elif self.key_model_name == "Llama":
      self.data_labels = [self.keyword_dict[self.cluster_topics[i]][0] for i in range(0, len(self.data))]
    return self.data_labels
    


  def get_repr_doc(self, cluster, n_doc=5):
    '''
    Returns representative documents for the specified cluster/topic

    Args:
    cluster : int
      Cluster/topic
    n_doc : int | default = 5
      Number of representative documents to be returned.

    Returns:
    cluster_doc_list : list
      List of representative documents for the specified `cluster`
    '''
    cluster_doc_list = []
    # Create list of documents belonging to 'cluster' and respective cluster probabilities
    for i in range(len(self.cluster_topics)):
      if self.cluster_topics[i] == cluster:
        cluster_doc_list.append((self.data[i], self.cluster_prob[i]))

    # Sort documents by cluster probabilities
    cluster_doc_list = sorted(cluster_doc_list, key=lambda x: x[1], reverse=True)

    # Return top n clusters
    return cluster_doc_list[:5]



  def get_cluster_topic_info(self):
    '''
    Returns a pandas dataframe containing cluster wise info with top `n_labels` labels
    '''
    # Create dataframe
    self.cluster_topic_info = pd.DataFrame()
    # List varables to store topic (cluster labels) and topic representation (topic name)
    topic = []
    representation = []
    rep_doc = []
    size_cluster = []

    for k,v in self.keyword_dict.items():
      # Cluster Id
      topic.append(k)
      # List of labels
      if "KeyBert" in self.keyword_model:
        key_model_name = "KeyBert"
        representation.append([r[0] for r in v])
      elif "Llama" in self.keyword_model:
        key_model_name = "Llama"
        # Since LLama returns only 1 label, append empty strings as remaining n_labels-1 labels
        c_label = [v[0]]
        for _ in range(self.n_labels-1):
          c_label.append(" ")
        representation.append(c_label)
      # Representative document
      cluster_repr_doc = self.get_repr_doc(cluster=k, n_doc=1)[0]
      rep_doc.append(cluster_repr_doc)
      # Cluster size
      size_cluster.append(len([1 for i in range(len(self.data)) if self.cluster_topics[i] == k]))
      
    self.cluster_topic_info['Topic'] = topic
    self.cluster_topic_info['Size'] = size_cluster
    self.cluster_topic_info[key_model_name] = representation
    self.cluster_topic_info["Representative Doc"] = rep_doc
    return self.cluster_topic_info



  def visualize(self, short_text, embeddings):
    '''
    Plots a scatterplot with data embeddings as coordinates.

    Args: 
    short_text : list
      Description of coordinate to be displayed on hover.
    embeddings : list
      List of 2D embeddings for visualization.

    Returns:
    None
    '''

    # Hover data
    hover_data = dict()
    hover_data['x'] = False
    hover_data['y'] = False
    hover_data['desc'] = short_text
    
    # Plot the scatter plot
    labels = self.get_labels()
    vis_dframe = pd.DataFrame()
    vis_dframe['x'] = embeddings[:, 0]
    vis_dframe['y'] = embeddings[:, 1]
    vis_dframe['labels'] = labels
    fig = px.scatter(vis_dframe, x='x', y='y', template="plotly_dark",
                      title="Document and Topics - " + list(self.keyword_model.keys())[0], color='labels',
                      hover_data=hover_data)

    # Update height
    fig.update_layout(
      autosize=False,
      width=1600,
      height=800,
    )
    
    fig.show()



  def get_cluster_labels(self, cluster):
    '''
    Returns a list of top `n_labels` labels for the specified cluster topic.

    Args: 
    cluster : int
      cluster topic as returned by clustering algorithm

    Returns:
    keyword_list : list
      List of labels obtained by fitting a keyword_model on data from `cluster`
    '''
    # Get list of top representative documents (default = 5)
    cluster_repr_doc = self.get_repr_doc(cluster=cluster)
    cluster_repr_doc = "- " + "\n- ".join([c[0] for c in cluster_repr_doc])
    cluster_repr_doc = cluster_repr_doc.strip()

    if "KeyBert" in self.keyword_model:
      if self.key_model_name == None:
        self.key_model_name = "KeyBert"

      # Get the KeyBert model
      model = self.keyword_model['KeyBert']
      
      # Extract keywords using the model
      self.keyword_list = model.extract_keywords(cluster_repr_doc,
                                                top_n=self.n_labels)
    elif "Llama" in self.keyword_model:
      if self.key_model_name == None:
        self.key_model_name = "Llama"

      # Get the generator (model pipeline) and the prompt template
      generator = self.keyword_model['Llama'][0]
      prompt = self.keyword_model['Llama'][1]

      # Replace [DOCUMENTS] in prompt with representative documents
      prompt = prompt.replace("[DOCUMENTS]", cluster_repr_doc)

      # Get cluster labels (for LLama-2 = 1)
      res = generator(prompt)
      self.keyword_list = [res[0]["generated_text"][len(prompt):].strip()]
    else:
      raise ValueError("Only KeyBert and Llama are supported for keyword generation")
    
    return self.keyword_list



  def fit_transform(self, data):
    '''
    Transforms the data and fits a TopicModel on the transformed data. 

    Args:
    data : list
      List of text data to be modelled
    
    Returns:
    None
    '''
    # Raise error if data is None
    if data == None:
      raise ValueError("Data cannot be None")
    else:
      self.data = data

    # Get embeddings using embedding model
    print("TopicModel - Creating embeddings...")
    self.embeddings = None
    if self.embedding_model == None:
      raise ValueError("Embedding model cannot be None")
    else:
      if self.embeddings == None:
        self.embeddings = self.embedding_model.encode(data, show_progress_bar=True)
      print("TopicModel - Embeddings created")

    # Reduce embeddings size
    print("TopicModel - Reducing embeddings...")
    self.reduced_embeddings = None
    if self.dimreduce_model == None:
      raise ValueError("Dimensionality reduction model cannot be None")
    else:
      if self.reduced_embeddings == None:
        self.reduced_embeddings = self.dimreduce_model.fit_transform(self.embeddings)
      print("TopicModel - Embeddings reduced...")

    # Perform clustering
    print("TopicModel - Clustering reduced embeddings...")
    self.cluster_topics = None
    if self.cluster_model == None:
      raise ValueError("Clustering model cannot be None")
    else:
      if self.cluster_topics == None:
        self.cluster_topics = self.cluster_model.fit_predict(self.reduced_embeddings)
        self.cluster_prob = self.cluster_model.probabilities_
      print("TopicModel - Clusters created")
    
    # Fit keyword model
    print("TopicModel - Generating labels...")
    self.keyword_dict = None
    if self.keyword_model == None:
      raise ValueError("Keyword model cannot be None")
    else:
      if self.keyword_dict == None:
        self.keyword_dict = defaultdict(list)
        # Iterate through each topic and get top `n_labels` labels for that cluster topic
        for cluster in np.unique(self.cluster_topics):
          self.keyword_dict[cluster] = self.get_cluster_labels(cluster)
    print("TopicModel - Keywords generated")

    return None