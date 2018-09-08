package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.store.io.VectorStoreTextWriter;
import com.gengoai.cache.AutoCalculatingLRUCache;
import com.gengoai.cache.Cache;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.logging.Loggable;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Serializable;
import java.util.*;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;

/**
 * The type Indexed file store.
 *
 * @author David B. Bracewell
 */
public final class DiskBasedVectorStore implements VectorStore, Serializable, Loggable {
   private static final long serialVersionUID = 1L;
   private final Map<String, Long> keyOffsets;
   private final File vectorFile;
   private transient final Cache<String, NDArray> vectorCache;
   private final int dimension;
   private final int cacheSize;


   private DiskBasedVectorStore(File vectorFile,
                                int cacheSize
                               ) throws IOException {
      this.vectorFile = vectorFile;
      this.keyOffsets = new HashMap<>();
      this.cacheSize = cacheSize;
      this.dimension = VectorStoreTextWriter.determineDimension(vectorFile);
      indexFile();
      vectorCache = new AutoCalculatingLRUCache<>(cacheSize, this::loadNDArray);

   }

   public static void main(String[] args) throws Exception {
      DiskBasedVectorStore.Builder builder = builder().location(new File("/home/ik/glove.6B.50d.txt"));
//      for (int i = 0; i < 100; i++) {
//         builder.add(StringUtils.randomHexString(10), NDArrayFactory.DENSE
//                                                         .create(NDArrayInitializer.rand, 50));
//      }
      VectorStore vs = builder.build();
      vs.keySet().forEach(key -> System.out.println(key + " : " + vs.get(key)));
   }

   /**
    * Builder builder.
    *
    * @return the builder
    */
   public static Builder builder() {
      return new Builder();
   }

   /**
    * Convenience method for loading an indexed vector store
    *
    * @param resource the resource containing the vectors
    * @return the vector store
    * @throws IOException Something went wrong reading the store
    */
   public static VectorStore read(Resource resource) throws IOException {
      return builder().location(resource.asFile().orElseThrow(IllegalStateException::new)).build();
   }

   @Override
   public boolean containsKey(String s) {
      return keyOffsets.containsKey(s);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(String s) {
      return vectorCache.get(s);
   }

   private void indexFile() throws IOException {
      File indexFile = VectorStoreTextWriter.indexFileFor(vectorFile);
      if (indexFile.exists()) {
         keyOffsets.putAll(VectorStoreTextWriter.readIndexFor(vectorFile));
      } else {
         keyOffsets.putAll(VectorStoreTextWriter.createIndexFor(vectorFile));
         VectorStoreTextWriter.writeIndexFor(vectorFile, keyOffsets);
      }
   }

   @Override
   public Iterator<NDArray> iterator() {
      return new Iterator<NDArray>() {
         private final Iterator<String> itr = keyOffsets.keySet().iterator();

         @Override
         public boolean hasNext() {
            return itr.hasNext();
         }

         @Override
         public NDArray next() {
            return vectorCache.get(itr.next());
         }
      };
   }

   @Override
   public Set<String> keySet() {
      return Collections.unmodifiableSet(keyOffsets.keySet());
   }

   private NDArray loadNDArray(String key) {
      if (keyOffsets.containsKey(key)) {
         try (RandomAccessFile raf = new RandomAccessFile(vectorFile, "r")) {
            long offset = keyOffsets.get(key);
            NDArray vector = NDArrayFactory.DENSE.zeros(1, dimension);
            raf.seek(offset);
            String line = raf.readLine();
            String[] parts = line.split("[ \t]+");
            for (int i = 1; i < parts.length; i++) {
               vector.set(i - 1, Double.parseDouble(parts[i]));
            }
            vector.setLabel(parts[0]);
            return vector;
         } catch (Exception e) {
            e.printStackTrace();
         }
      }
      return NDArrayFactory.SPARSE.zeros(dimension);
   }

   @Override
   public int size() {
      return keyOffsets.size();
   }

   @Override
   public VectorStoreBuilder toBuilder() {
      return new Builder().location(vectorFile)
                          .cacheSize(cacheSize);
   }

   /**
    * The type Builder.
    */
   public static class Builder extends VectorStoreBuilder {
      public static final String CACHE_SIZE = "CACHE_SIZE";
      public static final String LOCATION = "LOCATION";
      private VectorStoreTextWriter writer = null;

      /**
       * Instantiates a new Builder.
       */
      public Builder() {
         parameter(CACHE_SIZE, 5_000);
      }

      private void ensureWriter() {
         if (writer == null) {
            try {
               writer = new VectorStoreTextWriter(dimension(),
                                                  Resources.temporaryFile()
                                                           .asFile()
                                                           .orElseThrow(IllegalStateException::new));
            } catch (IOException e) {
               throw new RuntimeException(e);
            }
         }
      }

      @Override
      public VectorStoreBuilder add(String key, NDArray vector) {
         notNullOrBlank(key, "The key must not be null or blank");
         try {
            if (dimension() == -1) {
               dimension((int) vector.length());
            }
            checkArgument(dimension() == vector.length(),
                          () -> "Dimension mismatch. (" + dimension() + ") != (" + vector.length() + ")");
            ensureWriter();
            writer.write(key, vector);
         } catch (IOException e) {
            throw new RuntimeException(e);
         }
         return this;
      }

      @Override
      public VectorStore build() throws IOException {
         File location = parameterAs(LOCATION, File.class);
         File indexLocation = VectorStoreTextWriter.indexFileFor(location);
         if (writer != null) {
            writer.close();
            Resource vectors = Resources.fromFile(writer.getVectorFile());
            Resource index = Resources.fromFile(writer.getIndexFile());
            vectors.copy(Resources.fromFile(location));
            index.copy(Resources.fromFile(indexLocation));
         }
         writer = null;
         return new DiskBasedVectorStore(location, parameterAs(CACHE_SIZE, Integer.class));
      }

      /**
       * Cache size builder.
       *
       * @param size the size
       * @return the builder
       */
      public Builder cacheSize(int size) {
         parameter(CACHE_SIZE, size);
         return this;
      }

      /**
       * Location builder.
       *
       * @param location the location
       * @return the builder
       */
      public Builder location(File location) {
         parameter(LOCATION, location);
         return this;
      }
   }

}//END OF DiskBasedVectorStore
