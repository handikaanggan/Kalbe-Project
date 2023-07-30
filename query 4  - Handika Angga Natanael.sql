select sum (qty) as total_produk, "Product Name"
from "transaction" t 
left outer join product
on t.productid = product.productid 
group by "Product Name" 
order by total_produk desc
limit 1
