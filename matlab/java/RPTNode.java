public class RPTNode{

	private double split;
	private RPTNode left;
	private RPTNode right;
	private int[] indexes;
	
	public RPTNode(int[] indexes){
		this.indexes = indexes;
		this.left = null;
		this.right = null;
		this.split = 0;	
	}
	
	public void set_children(RPTNode left, RPTNode right, double split){
		this.left = left;
		this.right = right;
		this.split = split;
	}
	
	public int[] get_indexes(){
		return this.indexes;
	}
    
    public RPTNode get_left(){
        return this.left;
    }
    
    public RPTNode get_right(){
        return this.right;
    }
    
    public double get_split(){
        return this.split;
    }
	
}
