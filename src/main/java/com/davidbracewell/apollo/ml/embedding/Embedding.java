package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.linear.store.DefaultVectorStore;
import com.davidbracewell.apollo.linear.store.LSHVectorStore;
import com.davidbracewell.apollo.linear.store.VectorStore;
import com.davidbracewell.apollo.linear.store.VectorStoreBuilder;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.encoder.EncoderPair;
import com.davidbracewell.apollo.stat.measure.Measure;
import com.davidbracewell.apollo.stat.measure.Similarity;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.guava.common.primitives.Ints;
import com.davidbracewell.io.resource.Resource;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>A word/feature embedding model where words/features have been mapped into vectors.</p>
 *
 * @author David B. Bracewell
 */
public class Embedding implements Model, VectorStore<String>, Serializable {
   private static final long serialVersionUID = 1L;
   private VectorStore<String> vectorStore;

   /**
    * Instantiates a new Embedding.
    *
    * @param vectorStore the vector store
    */
   public Embedding(@NonNull VectorStore<String> vectorStore) {
      this.vectorStore = vectorStore;
   }

   /**
    * From vector store embedding.
    *
    * @param vectors the vectors
    * @return the embedding
    */
   public static Embedding fromVectorStore(@NonNull VectorStore<String> vectors) {
      return new Embedding(vectors);
   }

   public static Embedding fromWord2VecTextFile(@NonNull Resource in, boolean fastNearestNeighbors) throws IOException {
      List<String> lines = in.readLines();
      int firstRow = 1;
      Integer dimension = Ints.tryParse(lines.get(0).split("\\s+")[1]);
      if (dimension == null) {
         firstRow = 0;
         dimension = lines.get(0).trim().split("\\s+").length - 1;
      }
      VectorStoreBuilder<String> builder;
      if (fastNearestNeighbors) {
         builder = LSHVectorStore.<String>builder().signature("COSINE");
      } else {
         builder = DefaultVectorStore.builder();
      }
      builder.dimension(dimension);
      builder.measure(Similarity.Cosine);

      lines.stream()
           .skip(firstRow)
           .parallel()
           .map(line -> {
              NDArray v = NDArrayFactory.DEFAULT().zeros(builder.dimension());
              String[] parts = line.trim().split("\\s+");
              for (int vi = 1; vi < parts.length; vi++) {
                 v.set(vi - 1, Double.parseDouble(parts[vi]));
              }
              v.setLabel(parts[0].replace('_', ' '));
              return v;
           })
           .collect(Collectors.toList())
           .forEach(builder::add);
      return new Embedding(builder.build());
   }


   /**
    * Reads the model from the given resource
    *
    * @param modelResource the resource containing the serialized model
    * @return the deserialized model
    * @throws Exception Something went wrong reading the model
    */
   public static Embedding read(@NonNull Resource modelResource) throws Exception {
      Object o = modelResource.readObject();
      if (o instanceof Embedding) {
         return Cast.as(o);
      } else if (o instanceof VectorStore) {
         return Embedding.fromVectorStore(Cast.as(o));
      }
      throw new IllegalArgumentException(o.getClass() + " cannot be read in as an embedding");
   }

   public static SerializableSupplier<Embedding> supplier(final boolean word2vecFormat, @NonNull final Resource resource) {
      if (word2vecFormat) {
         return Unchecked.supplier(() -> Embedding.fromWord2VecTextFile(resource, false));
      }
      return Unchecked.supplier(() -> Embedding.read(resource));
   }


   public boolean contains(String word) {
      return vectorStore.containsKey(word);
   }

   @Override
   public boolean containsKey(String s) {
      return vectorStore.containsKey(s);
   }

   @Override
   public VectorStoreBuilder<String> toBuilder() {
      return vectorStore.toBuilder();
   }

   @Override
   public int dimension() {
      return vectorStore.dimension();
   }

   @Override
   public NDArray get(String s) {
      return vectorStore.get(s);
   }

   @Override
   public EncoderPair getEncoderPair() {
      return EncoderPair.NO_OPT;
   }

   @Override
   public Iterator<NDArray> iterator() {
      return vectorStore.iterator();
   }

   @Override
   public Collection<String> keys() {
      return vectorStore.keys();
   }

   @Override
   public List<NDArray> nearest(@NonNull NDArray v, int K) {
      return nearest(v, K, Double.NEGATIVE_INFINITY);
   }

   @Override
   public List<NDArray> nearest(NDArray query, double threshold) {
      return vectorStore.nearest(query, threshold);
   }

   @Override
   public Measure getQueryMeasure() {
      return vectorStore.getQueryMeasure();
   }

   @Override
   public List<NDArray> nearest(@NonNull NDArray v, int K, double threshold) {
      return vectorStore.nearest(v, K, threshold);
   }

   @Override
   public List<NDArray> nearest(NDArray query) {
      return vectorStore.nearest(query);
   }

   @Override
   public int size() {
      return vectorStore.size();
   }

}// END OF Embedding
