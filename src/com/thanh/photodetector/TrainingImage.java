package com.thanh.photodetector;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import android.net.Uri;

public class TrainingImage {
	private String path;
	private long tour_id;
	private Mat matrix;
	private Mat descriptors;
	private Uri uri;
	
	public TrainingImage(String image_path, long tour_item_id)
	{
		path = image_path;
		tour_id = tour_item_id;
		matrix = Imgcodecs.imread(image_path);
	}
	
	public void setPath(String newPath){
		path= newPath;
	}
	
	public void setTourID(long new_tour_id){
		if(new_tour_id>=0){
			tour_id = new_tour_id;
		}else{
			System.out.println("Tour ID must be non-negative");
		}
	}
	
	public void setDescriptors(Mat descrpt){
		descriptors=descrpt;
	}
	
	public Mat mat(){
		return matrix;
	}
	
	public String path(){
		return path;
	}
	
	public long tourID(){
		return tour_id;
	}
	
	public Mat descriptors(){
		return descriptors;
	}
	
}
