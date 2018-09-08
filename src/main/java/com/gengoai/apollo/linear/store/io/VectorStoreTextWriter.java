package com.gengoai.apollo.linear.store.io;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.io.Resources;
import com.gengoai.json.JsonWriter;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;

/**
 * @author David B. Bracewell
 */
public class VectorStoreTextWriter implements VectorStoreWriter {

   private final int dimension;
   private RandomAccessFile vectorWriter;
   private JsonWriter indexWriter;
   private long lastOffset = 0;

   public VectorStoreTextWriter(int dimension, File vectorFile) throws IOException {
      this.dimension = dimension;
      File indexFile = new File(vectorFile.getAbsolutePath() + ".idx.json.gz");
      this.vectorWriter = new RandomAccessFile(vectorFile, "rw");
      this.indexWriter = new JsonWriter(Resources.fromFile(indexFile).setIsCompressed(true));
   }

   @Override
   public void close() throws Exception {
      if (lastOffset > 0) {
         indexWriter.endObject();
         indexWriter.endDocument();
      }
      indexWriter.close();
      vectorWriter.close();
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public void write(String key, NDArray vector) throws IOException {
      notNullOrBlank(key, "The key must not be null or blank");
      checkArgument(dimension == vector.length(),
                    () -> "Dimension mismatch. (" + dimension + ") != (" + vector.length() + ")");
      StringBuilder cLine = new StringBuilder(key);
      for (int i = 0; i < vector.length(); i++) {
         cLine.append(" ").append(vector.get(i));
      }
      cLine.append("\n");
      if (lastOffset == 0) {
         indexWriter.beginDocument();
         indexWriter.property("dimension", dimension);
         indexWriter.beginObject("offsets");
      }
      vectorWriter.write(cLine.toString().getBytes());
      indexWriter.property(key, lastOffset);
      lastOffset = vectorWriter.getFilePointer();
   }
}//END OF VectorStoreTextWriter
